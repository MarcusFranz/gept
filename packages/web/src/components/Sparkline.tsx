// packages/web/src/components/Sparkline.tsx
import { createMemo } from 'solid-js';

interface SparklineProps {
  prices: number[];
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

  const points = createMemo(() => {
    const prices = props.prices;
    if (!prices || prices.length < 2) return null;

    const allValues = [...prices];
    if (props.predictedPrice != null) allValues.push(props.predictedPrice);

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    const w = width();
    const h = height();
    const xStep = (w - padding * 2) / (prices.length - 1);

    const pts = prices.map((price, i) => ({
      x: padding + i * xStep,
      y: padding + (1 - (price - min) / range) * (h - padding * 2),
    }));

    let predicted: { x: number; y: number } | null = null;
    if (props.predictedPrice != null) {
      predicted = {
        x: w - padding,
        y: padding + (1 - (props.predictedPrice - min) / range) * (h - padding * 2),
      };
    }

    return { pts, predicted, min, max, range };
  });

  const polyline = createMemo(() => {
    const p = points();
    if (!p) return '';
    return p.pts.map(pt => `${pt.x},${pt.y}`).join(' ');
  });

  const areaPath = createMemo(() => {
    const p = points();
    if (!p) return '';
    const h = height();
    const pts = p.pts;
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
                <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.25" />
                <stop offset="100%" stop-color="var(--accent)" stop-opacity="0" />
              </linearGradient>
            </defs>

            {/* Area fill */}
            <path d={areaPath()} fill={`url(#${gradientId})`} />

            {/* History line */}
            <polyline
              points={polyline()}
              fill="none"
              stroke="var(--accent)"
              stroke-width="1.5"
              stroke-linejoin="round"
              stroke-linecap="round"
            />

            {/* Current price dot */}
            {(() => {
              const p = points()!;
              const last = p.pts[p.pts.length - 1];
              return (
                <circle
                  cx={last.x}
                  cy={last.y}
                  r="2.5"
                  fill="var(--accent)"
                />
              );
            })()}

            {/* Predicted price dashed line */}
            {points()!.predicted && (() => {
              const p = points()!;
              const last = p.pts[p.pts.length - 1];
              const pred = p.predicted!;
              return (
                <line
                  x1={last.x}
                  y1={last.y}
                  x2={pred.x}
                  y2={pred.y}
                  stroke="var(--text-muted)"
                  stroke-width="1"
                  stroke-dasharray="3,2"
                  stroke-linecap="round"
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
