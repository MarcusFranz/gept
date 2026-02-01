// packages/web/src/components/Sparkline.tsx
import { createMemo } from 'solid-js';

interface SparklineProps {
  highs: number[];
  lows: number[];
  width?: number;
  height?: number;
  class?: string;
}

export function Sparkline(props: SparklineProps) {
  const width = () => props.width ?? 120;
  const height = () => props.height ?? 32;
  const padding = 2;

  const spreadGradientId = `spark-spread-${Math.random().toString(36).slice(2, 8)}`;

  const points = createMemo(() => {
    const highs = props.highs;
    const lows = props.lows;
    const len = Math.min(highs?.length ?? 0, lows?.length ?? 0);
    if (len < 2) return null;

    const allValues = [...highs.slice(0, len), ...lows.slice(0, len)];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    const w = width();
    const h = height();
    const xStep = (w - padding * 2) / (len - 1);
    const toY = (val: number) => padding + (1 - (val - min) / range) * (h - padding * 2);

    const highPts = highs.slice(0, len).map((v, i) => ({ x: padding + i * xStep, y: toY(v) }));
    const lowPts = lows.slice(0, len).map((v, i) => ({ x: padding + i * xStep, y: toY(v) }));

    return { highPts, lowPts };
  });

  // Spread band: highs forward, lows reversed
  const spreadPath = createMemo(() => {
    const p = points();
    if (!p) return '';
    const forward = p.highPts.map((pt, i) => (i === 0 ? `M${pt.x},${pt.y}` : `L${pt.x},${pt.y}`)).join(' ');
    const backward = [...p.lowPts].reverse().map(pt => `L${pt.x},${pt.y}`).join(' ');
    return `${forward} ${backward} Z`;
  });

  // High line polyline
  const highLine = createMemo(() => {
    const p = points();
    if (!p) return '';
    return p.highPts.map(pt => `${pt.x},${pt.y}`).join(' ');
  });

  // Low line polyline
  const lowLine = createMemo(() => {
    const p = points();
    if (!p) return '';
    return p.lowPts.map(pt => `${pt.x},${pt.y}`).join(' ');
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
              <linearGradient id={spreadGradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.15" />
                <stop offset="100%" stop-color="var(--accent)" stop-opacity="0.03" />
              </linearGradient>
            </defs>

            {/* Spread band fill between high and low */}
            <path d={spreadPath()} fill={`url(#${spreadGradientId})`} />

            {/* Low line */}
            <polyline
              points={lowLine()}
              fill="none"
              stroke="var(--accent)"
              stroke-width="1"
              stroke-opacity="0.4"
              stroke-linejoin="round"
              stroke-linecap="round"
            />

            {/* High line */}
            <polyline
              points={highLine()}
              fill="none"
              stroke="var(--accent)"
              stroke-width="1.5"
              stroke-linejoin="round"
              stroke-linecap="round"
            />

            {/* Current high price dot */}
            {(() => {
              const p = points()!;
              const last = p.highPts[p.highPts.length - 1];
              return (
                <circle
                  cx={last.x}
                  cy={last.y}
                  r="2"
                  fill="var(--accent)"
                />
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
