// packages/web/src/components/Sparkline.tsx
import { createMemo } from 'solid-js';

interface SparklineProps {
  highs: number[];
  lows: number[];
  loading?: boolean;
  width?: number;
  height?: number;
  class?: string;
}

export function Sparkline(props: SparklineProps) {
  const width = () => props.width ?? 120;
  const height = () => props.height ?? 32;
  const padding = 2;

  const placeholder = createMemo(() => {
    const w = width();
    const h = height();
    const innerW = w - padding * 2;
    const innerH = h - padding * 2;

    // Generic indeterminate "spark" shape. Coordinates are percentages.
    const xs = [0, 0.12, 0.22, 0.36, 0.5, 0.62, 0.78, 1];
    const ys = [0.72, 0.56, 0.66, 0.42, 0.54, 0.34, 0.48, 0.28];

    const pts = xs.map((x, i) => ({
      x: padding + x * innerW,
      y: padding + ys[i] * innerH,
    }));

    // Approximate path length (sum of segments) so we can animate "drawing" the stroke.
    let len = 0;
    for (let i = 1; i < pts.length; i += 1) {
      const dx = pts[i].x - pts[i - 1].x;
      const dy = pts[i].y - pts[i - 1].y;
      len += Math.hypot(dx, dy);
    }

    return {
      points: pts.map(pt => `${pt.x},${pt.y}`).join(' '),
      last: pts[pts.length - 1],
      len,
    };
  });

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
        {props.loading && !points() && (
          <>
            <polyline
              points={placeholder().points}
              class="sparkline-placeholder-base"
            />
            <polyline
              points={placeholder().points}
              class="sparkline-placeholder-draw"
              style={{ '--spark-dash': `${placeholder().len}px` }}
            />
            <circle
              cx={placeholder().last.x}
              cy={placeholder().last.y}
              r="2"
              class="sparkline-placeholder-dot"
            />
          </>
        )}

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

        .sparkline-placeholder-base {
          fill: none;
          stroke: color-mix(in srgb, var(--text-muted) 35%, transparent);
          stroke-width: 1.15;
          stroke-linejoin: round;
          stroke-linecap: round;
          opacity: 0.7;
        }

        .sparkline-placeholder-draw {
          fill: none;
          stroke: var(--accent);
          stroke-width: 1.75;
          stroke-linejoin: round;
          stroke-linecap: round;
          opacity: 0.85;
          stroke-dasharray: var(--spark-dash, 220px) var(--spark-dash, 220px);
          stroke-dashoffset: var(--spark-dash, 220px);
          animation: sparklinePlaceholderDraw 1.25s linear infinite;
        }

        .sparkline-placeholder-dot {
          fill: var(--accent);
          opacity: 0;
          animation: sparklinePlaceholderDot 1.25s linear infinite;
        }

        @keyframes sparklinePlaceholderDraw {
          0% {
            stroke-dashoffset: var(--spark-dash, 220px);
            opacity: 0;
          }
          10% {
            opacity: 0.85;
          }
          72% {
            stroke-dashoffset: 0;
            opacity: 0.85;
          }
          90% {
            opacity: 0;
          }
          100% {
            stroke-dashoffset: 0;
            opacity: 0;
          }
        }

        @keyframes sparklinePlaceholderDot {
          0%,
          60% {
            opacity: 0;
          }
          72% {
            opacity: 0.85;
          }
          90%,
          100% {
            opacity: 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .sparkline-placeholder-draw,
          .sparkline-placeholder-dot {
            animation: none;
          }
          .sparkline-placeholder-draw {
            stroke-dashoffset: 0;
            opacity: 0.7;
          }
          .sparkline-placeholder-dot {
            opacity: 0.7;
          }
        }
      `}</style>
    </>
  );
}
