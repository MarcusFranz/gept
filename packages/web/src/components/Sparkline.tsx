// packages/web/src/components/Sparkline.tsx
import { createMemo } from 'solid-js';

interface SparklineProps {
  highs: number[];
  lows: number[];
  predictedPrice?: number;
  width?: number;
  height?: number;
  class?: string;
}

export function Sparkline(props: SparklineProps) {
  const width = () => props.width ?? 120;
  const height = () => props.height ?? 32;
  const padding = 2;

  const gradientId = `spark-grad-${Math.random().toString(36).slice(2, 8)}`;
  const spreadGradientId = `spark-spread-${Math.random().toString(36).slice(2, 8)}`;

  const points = createMemo(() => {
    const highs = props.highs;
    const lows = props.lows;
    const len = Math.min(highs?.length ?? 0, lows?.length ?? 0);
    if (len < 2) return null;

    const mids = Array.from({ length: len }, (_, i) => (highs[i] + lows[i]) / 2);

    const allValues = [...highs.slice(0, len), ...lows.slice(0, len)];
    if (props.predictedPrice != null) allValues.push(props.predictedPrice);

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    const w = width();
    const h = height();
    // Reserve 15% of width for prediction projection when prediction exists
    const hasPred = props.predictedPrice != null;
    const historyWidth = hasPred ? (w - padding * 2) * 0.85 : (w - padding * 2);
    const xStep = historyWidth / (len - 1);

    const toY = (val: number) => padding + (1 - (val - min) / range) * (h - padding * 2);

    const highPts = highs.slice(0, len).map((v, i) => ({ x: padding + i * xStep, y: toY(v) }));
    const lowPts = lows.slice(0, len).map((v, i) => ({ x: padding + i * xStep, y: toY(v) }));
    const midPts = mids.map((v, i) => ({ x: padding + i * xStep, y: toY(v) }));

    let predicted: { x: number; y: number } | null = null;
    if (hasPred) {
      predicted = {
        x: w - padding,
        y: toY(props.predictedPrice!),
      };
    }

    const lastMid = mids[mids.length - 1];
    const predAbove = props.predictedPrice != null ? props.predictedPrice > lastMid : false;

    return { highPts, lowPts, midPts, predicted, predAbove };
  });

  // Spread band: path from highs forward then lows reversed
  const spreadPath = createMemo(() => {
    const p = points();
    if (!p) return '';
    const forward = p.highPts.map((pt, i) => (i === 0 ? `M${pt.x},${pt.y}` : `L${pt.x},${pt.y}`)).join(' ');
    const backward = [...p.lowPts].reverse().map(pt => `L${pt.x},${pt.y}`).join(' ');
    return `${forward} ${backward} Z`;
  });

  // Midpoint polyline
  const midline = createMemo(() => {
    const p = points();
    if (!p) return '';
    return p.midPts.map(pt => `${pt.x},${pt.y}`).join(' ');
  });

  // Area fill below midpoint line
  const areaPath = createMemo(() => {
    const p = points();
    if (!p) return '';
    const h = height();
    const pts = p.midPts;
    const first = pts[0];
    const last = pts[pts.length - 1];
    const line = pts.map(pt => `L${pt.x},${pt.y}`).join(' ');
    return `M${first.x},${h} ${line} L${last.x},${h} Z`;
  });

  return (
    <>
      <svg
        class={`sparkline ${props.class ?? ''}`}
        viewBox={`0 0 ${width()} ${height()}`}
        width={width()}
        height={height()}
        preserveAspectRatio="none"
      >
        {points() && (
          <>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.2" />
                <stop offset="100%" stop-color="var(--accent)" stop-opacity="0" />
              </linearGradient>
              <linearGradient id={spreadGradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.12" />
                <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.04" />
              </linearGradient>
            </defs>

            {/* Spread band (high-low range) */}
            <path d={spreadPath()} fill={`url(#${spreadGradientId})`} />

            {/* Area fill below midpoint */}
            <path d={areaPath()} fill={`url(#${gradientId})`} />

            {/* Midpoint line */}
            <polyline
              points={midline()}
              fill="none"
              stroke="var(--accent)"
              stroke-width="1.5"
              stroke-linejoin="round"
              stroke-linecap="round"
            />

            {/* Current price dot */}
            {(() => {
              const p = points()!;
              const last = p.midPts[p.midPts.length - 1];
              return (
                <circle
                  cx={last.x}
                  cy={last.y}
                  r="2.5"
                  fill="var(--accent)"
                />
              );
            })()}

            {/* Predicted price: dashed line projecting to the right + dot */}
            {points()!.predicted && (() => {
              const p = points()!;
              const last = p.midPts[p.midPts.length - 1];
              const pred = p.predicted!;
              const color = p.predAbove ? 'var(--success)' : 'var(--danger)';
              return (
                <>
                  <line
                    x1={last.x}
                    y1={last.y}
                    x2={pred.x}
                    y2={pred.y}
                    stroke={color}
                    stroke-width="1.5"
                    stroke-dasharray="3,2"
                    stroke-linecap="round"
                  />
                  <circle
                    cx={pred.x}
                    cy={pred.y}
                    r="2.5"
                    fill={color}
                  />
                </>
              );
            })()}
          </>
        )}
      </svg>

      <style>{`
        .sparkline {
          display: block;
          overflow: visible;
        }
      `}</style>
    </>
  );
}
