import { createSignal, type JSX } from 'solid-js';
import './Tooltip.css';

interface TooltipProps {
  text: string;
  children: JSX.Element;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
}

export default function Tooltip(props: TooltipProps) {
  const [visible, setVisible] = createSignal(false);
  const [coords, setCoords] = createSignal({ top: 0, left: 0 });
  let wrapperRef: HTMLSpanElement | undefined;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  const calculatePosition = () => {
    if (!wrapperRef) return;
    const rect = wrapperRef.getBoundingClientRect();
    const pos = props.position || 'top';

    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    let top = 0;
    let left = centerX;

    switch (pos) {
      case 'top':
        top = rect.top - 8;
        break;
      case 'bottom':
        top = rect.bottom + 8;
        break;
      case 'left':
        top = centerY;
        left = rect.left - 8;
        break;
      case 'right':
        top = centerY;
        left = rect.right + 8;
        break;
    }

    setCoords({ top, left });
  };

  const showTooltip = () => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      calculatePosition();
      setVisible(true);
    }, props.delay ?? 300);
  };

  const hideTooltip = () => {
    if (timeoutId) clearTimeout(timeoutId);
    setVisible(false);
  };

  const positionClass = () => props.position || 'top';

  return (
    <span
      ref={wrapperRef}
      class="tooltip-wrapper"
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {props.children}
      <span
        class={`tooltip-content tooltip-${positionClass()} ${visible() ? 'visible' : ''}`}
        style={{ top: `${coords().top}px`, left: `${coords().left}px` }}
      >
        {props.text}
      </span>
    </span>
  );
}

// Info icon component for use with tooltips
export function InfoIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      class="info-icon"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="6.5" />
      <path d="M8 7v4" />
      <circle cx="8" cy="5" r="0.5" fill="currentColor" />
    </svg>
  );
}
