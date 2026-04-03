# QuEL-3

## 概要

QuEL-3 は Qubex の backend familgy の一つ。

## backend adapter について

> [!NOTE]
> 対象ファイル
>
> - `src/qubex/measurement/adapters/quel3_backend_adapter.py`

Qubex の「論理的な測定リクエスト `MeasurementSchedule`」と QuEL-3 の「物理実行フォーマット `Quel3ExecutionPayload`」の間にある翻訳層。

- `MeasurementSchedule` は「各 target にどんな pulse を出すか」と「どの時刻に capture window を開くか」を保持
- `Quel3ExecutionPayload` は fixed timeline 形式で「各 instrument alias に対して、どの waveform をどこに置き、どこで capture するか」を保持

> [!IMPORTANT]
> Qubex の用語
>
> - target: Qubex の論理チャネル名。例: Q00, RQ00。
> - alias: QuEL-3/quelware 側の物理 instrument 名。
> - event: ある waveform をある時刻に置く命令。
> - capture window: ある時刻区間の取得窓。
> - fixed timeline: 1 つの target あるいは alias に対する event/capture の時系列セット。
> - MeasurementConfig: 何 shot 実行し、平均するか、時間積分するかを表す設定。

### `build_execution_request` がやっていること

- target ごとに alias をチェックする
  - `alias_map`

  ```py
  configured_alias = str(alias_map.get(target, "")).strip()
  if len(configured_alias) == 0:
      raise ValueError(
          "Missing QuEL-3 instrument alias mapping for "
          f"target `{target}`. Configure instruments before measurement."
      )
  instrument_bindings[target] = f"alias:{configured_alias}"
  ```

- 各 target の pulse sequence を `_create_waveform_events()` で waveform library と event 列に分解する

  ```py
  sequence = pulse_schedule.get_sequence(target, copy=False)
  target_type = self._experiment_system.get_target(target).type
  events, waveform_index = self._create_waveform_events(
      target_is_read=(target_type is TargetType.READ),
      sequence=sequence,
      waveform_name_by_shape_key=waveform_name_by_shape_key,
      waveform_library=waveform_library,
      waveform_index=waveform_index,
  )
  ```

  - waveform library:
    - 同じ shape と sampling period の波形は waveform library で共有される
  - `_create_waveform_events()` の中身（抜粋）

  ```py
  sampling_period_ns = waveform.sampling_period
  scale = waveform.scale
  shape = np.asarray(waveform.shape_values, dtype=np.complex128)
  if shape.size == 0:
      current_offset_ns += duration_ns
      continue

  # shape や sampling period を正規化する
  shape, sampling_period_ns = cls._normalize_waveform_for_target(
      target_is_read=target_is_read,
      shape=shape,
      sampling_period_ns=sampling_period_ns,
  )

  # waveform (のハッシュ) と sampling period から key を作り、すでに名前が与えられているか確認する
  shape_key = (
      waveform.shape_hash,
      round(float(sampling_period_ns) * 1e6),
  )
  waveform_name = waveform_name_by_shape_key.get(shape_key)
  if waveform_name is None:
      # 名前がないなら新しく名前を付ける
      waveform_name = f"waveform_{waveform_index:04d}"
      waveform_index += 1
      # ライブラリに波形を登録する
      waveform_library[waveform_name] = Quel3Waveform(
          iq_array=shape,
          sampling_period_ns=sampling_period_ns,
      )
      waveform_name_by_shape_key[shape_key] = waveform_name

  # イベントを追加する
  events.append(
      Quel3WaveformEvent(
          waveform_name=waveform_name,
          start_offset_ns=current_offset_ns,
          gain=scale,
          phase_offset_deg=math.degrees(waveform.phase),
      )
  )
  current_offset_ns += duration_ns
  ```

- readout target は `_normalize_waveform_for_target` で 0.8 ns grid に正規化される
  - control 側は backend 共通の　0.4 ns で動くが、 readout は QuEL-3 の readout sampling grid にそろえる必要がある
- capture schedule は target ごとに start_time 順に並べて `Quel3CaptureWindow` に変換する

  ```py
  captures = sorted(
      channel_captures.get(target, []), key=lambda c: c.start_time
  )
  capture_delay_ns = self._resolve_capture_delay_ns(
      target=target,
      target_type=target_type,
  )
  capture_windows = tuple(
      Quel3CaptureWindow(
          name=f"{target}:{index}",
          start_offset_ns=capture.start_time + capture_delay_ns,
          length_ns=capture.duration,
      )
      for index, capture in enumerate(captures)
  )
  timeline_length_ns = pulse_schedule.duration
  if len(capture_windows) > 0:
      timeline_length_ns = max(
          timeline_length_ns,
          max(
              window.start_offset_ns + window.length_ns
              for window in capture_windows
          ),
      )
  fixed_timelines[target] = Quel3FixedTimeline(
      events=events,
      capture_windows=capture_windows,
      length_ns=timeline_length_ns,
  )
  try:
      output_target_labels_by_target[target] = str(
          self._experiment_system.resolve_qubit_label(target)
      )
  except ValueError:
      output_target_labels_by_target[target] = str(
          target_registry.measurement_output_label(target)
      )
  ```

- readout target なら mux ごとの capture delay を足す
- `MeasurementConfig` は QuEL-3 の capture mode を決める
- Payload を作成する

  ```py
  payload = Quel3ExecutionPayload(
      waveform_library=waveform_library,
      fixed_timelines=fixed_timelines,
      n_iterations=n_iterations,
      shot_interval_ns=math.ceil(config.shot_interval),
      capture_mode=capture_mode,
      instrument_bindings=instrument_bindings,
  )
  ```

- 最後に `self._capture_targets_by_alias` を作成する
  - 同じ alias に複数 target がぶら下がったときに、返り値の 1 番目がどの target で、 2 番目がどの target かを後で戻すための対応表である

（例）

- RQ00 に 0.4 ns 間隔の readout pulse `[1,3,5,7]` を置くと、 adapter はこれを 0.8 ns grid にダウンサンプルして `[2,6]` にする
- event は「`waveform_xxxx` を `start_offset_ns=0.0` に置く」という形になる
  - たとえば、 capture は `RQ00:0` を `start_offset_ns=0.4` に置く、という形になる

### `build_measurement_result` がやっていること

> [!IMPORTANT]
> **TODO** 書く！
