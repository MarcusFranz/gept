import { createSignal, Show } from 'solid-js';
import type { Recommendation } from '../lib/types';
import { formatGold, formatPercent } from '../lib/types';
import FeedbackButton from './FeedbackButton';
import Tooltip, { InfoIcon } from './Tooltip';
import './FlipCard.css';

interface FlipCardProps {
  recommendation: Recommendation;
  onTrack?: () => void;
  onReport?: () => void;
  isTracking?: boolean;
}

export default function FlipCard(props: FlipCardProps) {
  const [expanded, setExpanded] = createSignal(false);

  const roi = () => {
    const { buyPrice, sellPrice, quantity } = props.recommendation;
    return (sellPrice - buyPrice) * quantity / (buyPrice * quantity);
  };

  const trendIcon = () => {
    switch (props.recommendation.trend) {
      case 'up': return '↑';
      case 'down': return '↓';
      default: return '→';
    }
  };

  const trendClass = () => {
    switch (props.recommendation.trend) {
      case 'up': return 'text-success';
      case 'down': return 'text-danger';
      default: return 'text-secondary';
    }
  };

  const confidenceClass = () => {
    const conf = props.recommendation.confidence;
    if (conf >= 0.8) return 'badge-success';
    if (conf >= 0.6) return 'badge-warning';
    return 'badge-danger';
  };

  return (
    <div class="flip-card card">
      <div class="flip-header">
        <div class="flip-item">
          <h3 class="flip-item-name">{props.recommendation.item}</h3>
          <span class={`flip-trend ${trendClass()}`}>
            {trendIcon()} {props.recommendation.trend}
          </span>
        </div>
        <div class="flip-header-right">
          <FeedbackButton recommendation={props.recommendation} compact />
          <Tooltip text="Model confidence in this recommendation. Higher = more certain." position="bottom">
            <span class={`badge ${confidenceClass()}`}>
              {formatPercent(props.recommendation.confidence)} conf
            </span>
          </Tooltip>
        </div>
      </div>

      <div class="flip-prices">
        <div class="flip-price">
          <span class="flip-price-label">Buy</span>
          <span class="flip-price-value font-mono">{formatGold(props.recommendation.buyPrice, 'gp')}</span>
        </div>
        <div class="flip-price-arrow">→</div>
        <div class="flip-price">
          <span class="flip-price-label">Sell</span>
          <span class="flip-price-value font-mono">{formatGold(props.recommendation.sellPrice, 'gp')}</span>
        </div>
      </div>

      <div class="flip-stats grid-3">
        <div class="flip-stat">
          <span class="flip-stat-value text-success font-mono">
            +{formatGold(props.recommendation.expectedProfit)}
          </span>
          <span class="flip-stat-label">Profit</span>
        </div>
        <div class="flip-stat">
          <span class="flip-stat-value font-mono">{formatGold(props.recommendation.capitalRequired)}</span>
          <span class="flip-stat-label">Capital</span>
        </div>
        <div class="flip-stat">
          <span class="flip-stat-value font-mono">{props.recommendation.quantity.toLocaleString()}</span>
          <span class="flip-stat-label">Quantity</span>
        </div>
      </div>

      <div class="flip-roi">
        <span class="flip-roi-label">ROI</span>
        <span class={`flip-roi-value font-mono ${roi() > 0 ? 'text-success' : 'text-danger'}`}>
          {formatPercent(roi())}
        </span>
      </div>

      <Show when={expanded()}>
        <div class="flip-details">
          <div class="flip-detail-row">
            <span class="flip-detail-label">
              Fill Probability
              <Tooltip text="Likelihood your buy/sell orders will complete based on trading volume" position="top">
                <InfoIcon />
              </Tooltip>
            </span>
            <span class="flip-detail-value">{formatPercent(props.recommendation.fillProbability)}</span>
          </div>
          <div class="flip-detail-row">
            <span class="flip-detail-label">
              Expected Time
              <Tooltip text="Estimated time to complete this flip based on market activity" position="top">
                <InfoIcon />
              </Tooltip>
            </span>
            <span class="flip-detail-value">{props.recommendation.expectedHours.toFixed(1)}h</span>
          </div>
          <div class="flip-detail-row">
            <span class="flip-detail-label">24h Volume</span>
            <span class="flip-detail-value">{props.recommendation.volume24h.toLocaleString()}</span>
          </div>
          <Show when={props.recommendation.isMultiLimitStrategy}>
            <div class="flip-detail-row">
              <span class="flip-detail-label">Strategy</span>
              <span class="flip-detail-value badge badge-info">Multi-Limit</span>
            </div>
          </Show>
          <div class="flip-reason">
            <p class="text-sm text-secondary">{props.recommendation.reason}</p>
          </div>
        </div>
      </Show>

      <div class="flip-actions">
        <button
          class="btn btn-ghost btn-sm"
          onClick={() => setExpanded(!expanded())}
        >
          {expanded() ? 'Less' : 'More'}
        </button>
        <div class="flip-actions-right">
          <Show when={props.onReport}>
            <button class="btn btn-secondary btn-sm" onClick={props.onReport}>
              Report
            </button>
          </Show>
          <Show when={props.onTrack}>
            <button
              class="btn btn-primary btn-sm"
              onClick={props.onTrack}
              disabled={props.isTracking}
            >
              {props.isTracking ? 'Tracking...' : 'Track Trade'}
            </button>
          </Show>
        </div>
      </div>
    </div>
  );
}
