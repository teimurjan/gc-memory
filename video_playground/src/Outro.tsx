import {AbsoluteFill, interpolate, spring, useVideoConfig} from "remotion";
import {
  BASELINE,
  BG,
  FONT_MONO,
  LETHE,
  TEXT,
  TEXT_DIM,
  TEXT_FAINT,
} from "./theme";
import type {RunData} from "./types";

type Props = {
  frame: number;
  durationInFrames: number;
  run: RunData;
};

export const Outro: React.FC<Props> = ({frame, durationInFrames, run}) => {
  const {fps} = useVideoConfig();
  const alpha = interpolate(
    frame,
    [0, 12, durationInFrames - 10, durationInFrames],
    [0, 1, 1, 0.9],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"},
  );

  const {baselineNdcg, lethNdcg, deltaPct} = run.meta.headline!;

  const labelS = spring({
    frame,
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 18,
  });
  const labelO = interpolate(labelS, [0, 1], [0, 1]);

  const numbersS = spring({
    frame: Math.max(0, frame - 18),
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 22,
  });
  const numbersO = interpolate(numbersS, [0, 1], [0, 1]);
  const numbersTy = interpolate(numbersS, [0, 1], [16, 0]);

  const pctS = spring({
    frame: Math.max(0, frame - 50),
    fps,
    config: {damping: 14, stiffness: 90},
  });
  const pctO = interpolate(pctS, [0, 1], [0, 1]);
  const pctScale = 0.8 + pctS * 0.2;

  const closingS = spring({
    frame: Math.max(0, frame - 100),
    fps,
    config: {damping: 200, stiffness: 200, mass: 0.6},
    durationInFrames: 18,
  });
  const closingO = interpolate(closingS, [0, 1], [0, 1]);

  return (
    <AbsoluteFill
      style={{
        background: BG,
        color: TEXT,
        fontFamily: FONT_MONO,
        opacity: alpha,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 36,
        padding: 60,
      }}
    >
      <div
        style={{
          fontSize: 22,
          color: TEXT_DIM,
          opacity: labelO,
          letterSpacing: 1,
        }}
      >
        LongMemEval-S
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 40,
          opacity: numbersO,
          transform: `translateY(${numbersTy}px)`,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        <span style={{fontSize: 72, color: BASELINE, fontWeight: 400}}>
          {baselineNdcg.toFixed(2)}
        </span>
        <span style={{fontSize: 44, color: LETHE}}>→</span>
        <span style={{fontSize: 96, color: LETHE, fontWeight: 600, letterSpacing: -2}}>
          {lethNdcg.toFixed(2)}
        </span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: 18,
          opacity: pctO,
          transform: `scale(${pctScale})`,
          marginTop: 8,
        }}
      >
        <span
          style={{
            fontSize: 168,
            fontWeight: 600,
            color: LETHE,
            lineHeight: 1,
            letterSpacing: -4,
          }}
        >
          +{Math.round(deltaPct)}%
        </span>
        <span style={{fontSize: 28, color: TEXT_DIM}}>NDCG</span>
      </div>

      <div
        style={{
          fontSize: 26,
          color: TEXT_FAINT,
          opacity: closingO,
          marginTop: 14,
        }}
      >
        no retraining
      </div>
    </AbsoluteFill>
  );
};
