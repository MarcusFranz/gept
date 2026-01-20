import { createSignal, createEffect, onCleanup, Show, For, untrack } from 'solid-js';
import type { Recommendation, UpdateRecommendation, SSEMessage } from '../lib/types';
import { formatGold, formatPercent } from '../lib/types';
import { addToast } from './ToastContainer';
import Tooltip, { InfoIcon } from './Tooltip';

// Styles
import './OrderGrid/OrderGrid.css';

// Extracted types and utilities
import type { SlotSettings, OrderState } from './OrderGrid/types';
import {
  MAX_SLOTS,
  TIER_LIMITS,
  STORAGE_KEY,
  USER_CAPITAL_KEY,
  DEFAULT_SETTINGS_KEY,
  PLAN_KEY,
  defaultSlotSettings,
} from './OrderGrid/types';
import {
  formatRsNumber,
  formatRsPrice,
  getPlanTier,
  loadUserCapital,
  loadDefaultSettings,
  loadSavedSlots,
  saveSlots,
  saveUserCapital,
  saveDefaultSettings,
  itemImageUrl,
  fillPercent,
  sellPercent,
  slotSettingsChanged,
} from './OrderGrid/utils';
import { useSSE } from './OrderGrid/useSSE';
import SignInModal from './OrderGrid/SignInModal';

interface OrderGridProps {
  isSignedIn?: boolean;
}

export default function OrderGrid(props: OrderGridProps) {
  const [slots, setSlots] = createSignal<OrderState[]>(loadSavedSlots());
  const [recommendations, setRecommendations] = createSignal<Recommendation[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [recIndex, setRecIndex] = createSignal(0);
  const [confirmingCancel, setConfirmingCancel] = createSignal<number | null>(null);
  const [draggingSlot, setDraggingSlot] = createSignal<number | null>(null);
  const [fineControlSlot, setFineControlSlot] = createSignal<number | null>(null);
  const [showInfoSlot, setShowInfoSlot] = createSignal<number | null>(null);
  const [hoverData, setHoverData] = createSignal<{ slotIndex: number; x: number; price: number; hoursAgo: number } | null>(null);
  const [tier, setTier] = createSignal<'free' | 'paid'>(getPlanTier());
  const [skippedHistory, setSkippedHistory] = createSignal<Recommendation[]>([]);
  const isSignedIn = () => props.isSignedIn ?? false;
  const [showSignInModal, setShowSignInModal] = createSignal(false);
  const [pendingOrderSlot, setPendingOrderSlot] = createSignal<number | null>(null);
  const [searchSlot, setSearchSlot] = createSignal<number | null>(null);
  const [searchQuery, setSearchQuery] = createSignal('');
  const [searchResults, setSearchResults] = createSignal<Array<{ id: number; name: string; icon: string }>>([]);
  const [searchLoading, setSearchLoading] = createSignal(false);

  // SSE connection state for real-time alerts
  const [sseConnected, setSseConnected] = createSignal(false);
  const [sseReconnectAttempts, setSseReconnectAttempts] = createSignal(0);
  let eventSource: EventSource | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // User capital and default settings
  const [userCapital, setUserCapital] = createSignal(loadUserCapital());
  const [globalDefaults, setGlobalDefaults] = createSignal<SlotSettings>(loadDefaultSettings());
  const [capitalInput, setCapitalInput] = createSignal('');

  // Track pending (uncommitted) settings changes
  const [pendingCapital, setPendingCapital] = createSignal<number | null>(null);
  const [pendingSettings, setPendingSettings] = createSignal<Partial<SlotSettings> | null>(null);

  // Check if there are any pending changes
  const hasPendingChanges = () => pendingCapital() !== null || pendingSettings() !== null;

  // Update capital (pending - not saved yet)
  const updateCapital = (value: number) => {
    setPendingCapital(value);
    setUserCapital(value); // Update display immediately
  };

  // Update settings (pending - not saved yet)
  const updateGlobalDefault = (key: keyof SlotSettings, value: string) => {
    setPendingSettings(prev => ({ ...prev, [key]: value }));
    setGlobalDefaults(prev => ({ ...prev, [key]: value })); // Update display immediately
  };

  // Confirm and apply pending changes
  // - Default settings changes: Just save for future cards, DON'T refresh recommendations
  // - Capital changes: Save AND refresh recommendations with new capital
  const confirmSettingsChanges = async () => {
    const capital = pendingCapital();
    const settings = pendingSettings();

    // Build the update payload
    const updates: Record<string, unknown> = {};
    if (capital !== null) updates.capital = capital;
    if (settings) Object.assign(updates, settings);

    if (Object.keys(updates).length === 0) return;

    // Save to localStorage
    if (capital !== null) {
      try { localStorage.setItem(USER_CAPITAL_KEY, capital.toString()); } catch {}
    }
    if (settings) {
      try { localStorage.setItem(DEFAULT_SETTINGS_KEY, JSON.stringify(globalDefaults())); } catch {}
    }

    // Save to server
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates)
    }).catch(() => {});

    // Clear pending state
    setPendingCapital(null);
    setPendingSettings(null);

    // Only refresh recommendations if capital changed
    // Default settings changes only apply to future cards (via slot settings inheritance)
    if (capital !== null) {
      setBrowseIndex(0);
      setSkippedHistory([]);

      // Use each slot's OWN settings for fetching, not global defaults
      // Capital affects what items are affordable, but style/risk/margin are per-slot
      const currentSettings = globalDefaults();
      const params = new URLSearchParams({
        capital: userCapital().toString(),
        style: currentSettings.style,
        risk: currentSettings.risk,
        margin: currentSettings.margin,
        fresh: '1' // Bypass cache after settings change
      });

      try {
        const response = await fetch(`/api/recommendations?${params}`);
        const data = await response.json();
        if (data.success && data.data) {
          setRecommendations(data.data);
          setSlots(prev => {
            let usedFirst = false;
            return prev.map(slot => {
              if (slot.isOrdered) return slot;
              if (!usedFirst && data.data.length > 0) {
                usedFirst = true;
                setBrowseIndex(1);
                return { ...slot, recommendation: data.data[0], isCustomItem: false, originalRecommendation: null };
              }
              return { ...slot, recommendation: null, isCustomItem: false, originalRecommendation: null };
            });
          });
        }
      } catch {}
    }

    // Close the panel
    setShowStatsPanel(false);
  };

  // Discard pending changes
  const discardSettingsChanges = () => {
    // Revert capital
    if (pendingCapital() !== null) {
      const savedCapital = loadUserCapital();
      setUserCapital(savedCapital);
    }
    // Revert settings
    if (pendingSettings() !== null) {
      const savedSettings = loadDefaultSettings();
      setGlobalDefaults(savedSettings);
    }
    // Clear pending state
    setPendingCapital(null);
    setPendingSettings(null);
  };

  const maxSlots = () => TIER_LIMITS[tier()];

  // Count active orders
  const activeOrderCount = () => slots().filter(s => s.isOrdered).length;

  // Visible slots = active orders + 1 recommendation slot (capped at tier limit)
  const visibleSlots = () => Math.min(activeOrderCount() + 1, maxSlots());

  // Dropdown panels
  const [showStatsPanel, setShowStatsPanel] = createSignal(false);
  const [showHistoryPanel, setShowHistoryPanel] = createSignal(false);

  // Computed stats for active trades
  const tradeStats = () => {
    const activeSlots = slots().filter(s => s.isOrdered && s.recommendation);
    const totalCapital = activeSlots.reduce((sum, s) => sum + (s.recommendation?.capitalRequired || 0), 0);
    const expectedProfit = activeSlots.reduce((sum, s) => sum + (s.recommendation?.expectedProfit || 0), 0);
    const buyingCount = activeSlots.filter(s => s.phase === 'buying').length;
    const sellingCount = activeSlots.filter(s => s.phase === 'selling').length;
    const avgFillProgress = activeSlots.length > 0
      ? activeSlots.reduce((sum, s) => {
          const progress = s.phase === 'buying'
            ? (s.filled / s.quantity)
            : (s.sold / s.filled);
          return sum + (isNaN(progress) ? 0 : progress);
        }, 0) / activeSlots.length
      : 0;
    return { totalCapital, expectedProfit, buyingCount, sellingCount, avgFillProgress, activeCount: activeSlots.length };
  };

  const handleProgressInteraction = (slotIndex: number, e: MouseEvent) => {
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = x / rect.width;
    const slot = slots()[slotIndex];
    const newFilled = Math.round(slot.quantity * percent);
    updateFilled(slotIndex, newFilled);
  };

  const handleProgressMouseDown = (slotIndex: number, e: MouseEvent) => {
    e.preventDefault(); // Prevent text selection
    setDraggingSlot(slotIndex);

    // Capture quantity at drag start to avoid reactive reads during drag
    const quantity = untrack(() => slots()[slotIndex]?.quantity) || 0;
    const progressEl = e.currentTarget as HTMLElement;

    // Initial update
    const rect = progressEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = x / rect.width;
    updateFilled(slotIndex, Math.round(quantity * percent));

    const handleMouseMove = (moveEvent: MouseEvent) => {
      moveEvent.preventDefault();
      const currentRect = progressEl.getBoundingClientRect();
      const moveX = Math.max(0, Math.min(moveEvent.clientX - currentRect.left, currentRect.width));
      const movePercent = moveX / currentRect.width;
      updateFilled(slotIndex, Math.round(quantity * movePercent));
    };

    const handleMouseUp = () => {
      setDraggingSlot(null);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const incrementFilled = (slotIndex: number, amount: number) => {
    const slot = slots()[slotIndex];
    const newFilled = Math.max(0, Math.min(slot.quantity, slot.filled + amount));
    updateFilled(slotIndex, newFilled);
  };

  // Selling phase progress handlers
  const handleSellProgressInteraction = (slotIndex: number, e: MouseEvent) => {
    const target = e.currentTarget as HTMLElement;
    const rect = target.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = x / rect.width;
    const slot = slots()[slotIndex];
    const newSold = Math.round(slot.filled * percent);
    updateSold(slotIndex, newSold);
  };

  const handleSellProgressMouseDown = (slotIndex: number, e: MouseEvent) => {
    e.preventDefault(); // Prevent text selection
    setDraggingSlot(slotIndex);

    // Capture filled at drag start to avoid reactive reads during drag
    const filled = untrack(() => slots()[slotIndex]?.filled) || 0;
    const progressEl = e.currentTarget as HTMLElement;

    // Initial update
    const rect = progressEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = x / rect.width;
    updateSold(slotIndex, Math.round(filled * percent));

    const handleMouseMove = (moveEvent: MouseEvent) => {
      moveEvent.preventDefault();
      const currentRect = progressEl.getBoundingClientRect();
      const moveX = Math.max(0, Math.min(moveEvent.clientX - currentRect.left, currentRect.width));
      const movePercent = moveX / currentRect.width;
      updateSold(slotIndex, Math.round(filled * movePercent));
    };

    const handleMouseUp = () => {
      setDraggingSlot(null);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const closeFineControl = () => {
    setFineControlSlot(null);
  };

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      // Use untrack to prevent this from creating reactive dependencies
      const currentSlots = untrack(() => slots());
      const activeItemIds = currentSlots
        .filter(s => s.isOrdered && s.recommendation)
        .map(s => s.recommendation!.itemId);

      // Build params with settings (untracked to prevent infinite loops)
      const settings = untrack(() => globalDefaults());
      const capital = untrack(() => userCapital());
      const params = new URLSearchParams({
        capital: capital.toString(),
        style: settings.style,
        risk: settings.risk,
        margin: settings.margin
      });

      if (activeItemIds.length > 0) {
        params.append('exclude', activeItemIds.join(','));
      }

      const response = await fetch(`/api/recommendations?${params}`);
      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch recommendations');
      }

      const recs = data.data || [];
      setRecommendations(recs);

      // Build set of active item IDs for filtering
      const activeItemIdSet = new Set(activeItemIds);

      // Preserve recommendations only for ORDERED slots, replace all others with fresh API data
      let recsUsedFromFetch = 0;
      setSlots(prev => {
        let recIdx = 0;
        return prev.map((slot) => {
          // Keep existing recommendation only if slot is actively ordered
          if (slot.isOrdered && slot.recommendation) {
            return { ...slot, settings: slot.settings || { ...defaultSlotSettings } };
          }
          // Find next rec that's not in active trades
          let rec = null;
          while (recIdx < recs.length) {
            if (!activeItemIdSet.has(recs[recIdx].itemId)) {
              rec = recs[recIdx];
              recIdx++;
              recsUsedFromFetch++;
              break;
            }
            recIdx++;
          }
          return {
            ...slot,
            recommendation: rec,
            isCustomItem: false, // Reset custom item flag for fresh recs
            settings: slot.settings || { ...defaultSlotSettings }
          };
        });
      });
      // Set recIndex to point to the next available recommendation
      setRecIndex(recsUsedFromFetch);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    fetchRecommendations();
  });


  // Load user settings from server on mount (sync with database)
  createEffect(() => {
    fetch('/api/settings')
      .then(res => res.json())
      .then(data => {
        if (data.success && data.data?.user) {
          const user = data.data.user;
          // Update local state with server values
          if (user.capital) {
            setUserCapital(user.capital);
            localStorage.setItem(USER_CAPITAL_KEY, user.capital.toString());
          }
          if (user.style || user.risk || user.margin) {
            const serverSettings = {
              style: user.style || globalDefaults().style,
              risk: user.risk || globalDefaults().risk,
              margin: user.margin || globalDefaults().margin
            };
            setGlobalDefaults(serverSettings);
            localStorage.setItem(DEFAULT_SETTINGS_KEY, JSON.stringify(serverSettings));
          }
        }
      })
      .catch(() => {
        // Ignore - use localStorage values as fallback
      });
  });

  // Preload the logo image to prevent flicker
  createEffect(() => {
    const img = new Image();
    img.src = '/images/logo-icon.png';
  });

  // Listen for plan changes
  createEffect(() => {
    const handlePlanChange = () => setTier(getPlanTier());
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === PLAN_KEY) handlePlanChange();
    };
    window.addEventListener('planchange', handlePlanChange);
    window.addEventListener('storage', handleStorageChange);
    onCleanup(() => {
      window.removeEventListener('planchange', handlePlanChange);
      window.removeEventListener('storage', handleStorageChange);
    });
  });

  // Persist slots to localStorage when they change (only if signed in)
  createEffect(() => {
    const currentSlots = slots();
    const signedIn = isSignedIn();

    if (!signedIn) {
      // Clear saved data if not signed in
      localStorage.removeItem(STORAGE_KEY);
      return;
    }

    // Only save slots that have active orders
    const toSave = currentSlots.map(slot => ({
      isOrdered: slot.isOrdered,
      phase: slot.phase,
      quantity: slot.quantity,
      filled: slot.filled,
      sold: slot.sold,
      recommendation: slot.recommendation,
      settings: slot.settings
      // Don't persist showSettings, pendingUpdate, lastCheckedAt
    }));
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    } catch {
      // Ignore localStorage errors (quota exceeded, private browsing, etc.)
    }
  });

  // SSE connection management for real-time alerts
  const connectSSE = () => {
    if (eventSource) {
      eventSource.close();
    }

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    // Don't connect if not signed in or tab is hidden
    if (!isSignedIn() || document.hidden) {
      return;
    }

    eventSource = new EventSource('/api/events/stream');

    eventSource.onopen = () => {
      setSseConnected(true);
      setSseReconnectAttempts(0);
    };

    eventSource.onmessage = (event) => {
      try {
        const message: SSEMessage = JSON.parse(event.data);

        if (message.type === 'ALERT' && message.data) {
          processUpdates([message.data]);
        }
        // CONNECTED and HEARTBEAT messages don't need handling
      } catch (err) {
        console.error('Failed to parse SSE message:', err);
      }
    };

    eventSource.onerror = () => {
      setSseConnected(false);
      eventSource?.close();
      eventSource = null;

      // Exponential backoff reconnection: 1s, 2s, 4s, 8s, 16s, 30s max
      const attempts = sseReconnectAttempts();
      const delay = Math.min(1000 * Math.pow(2, attempts), 30000);
      setSseReconnectAttempts(attempts + 1);

      // Schedule reconnect if tab is visible
      if (!document.hidden) {
        reconnectTimer = setTimeout(connectSSE, delay);
      }
    };
  };

  const disconnectSSE = () => {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    setSseConnected(false);
  };

  // Process incoming update recommendations
  const processUpdates = (updates: UpdateRecommendation[]) => {
    updates.forEach(update => {
      // Skip HOLD updates - they don't need UI
      if (update.type === 'HOLD') return;

      // Find the slot for this trade
      const slotIndex = slots().findIndex(
        s => s.isOrdered && s.recommendation?.id === update.tradeId
      );

      if (slotIndex === -1) return;

      const slot = slots()[slotIndex];

      // Don't overwrite existing pending update
      if (slot.pendingUpdate) return;

      // Set the pending update on the slot
      setSlots(prev => prev.map((s, i) => {
        if (i !== slotIndex) return s;
        return {
          ...s,
          pendingUpdate: update,
          lastCheckedAt: new Date().toISOString()
        };
      }));

      // Show toast notification
      const typeLabels: Record<string, string> = {
        'SWITCH_ITEM': 'Switch Recommended',
        'SELL_NOW': 'Sell Now Recommended',
        'ADJUST_PRICE': 'Price Adjustment'
      };

      addToast({
        type: 'warning',
        title: typeLabels[update.type] || 'Update Available',
        message: `${slot.recommendation?.item}: ${update.reason}`,
        duration: 8000,
        action: {
          label: 'View',
          onClick: () => {
            // Scroll to the relevant card
            const cardEl = document.querySelector(`[data-slot-index="${slotIndex}"]`);
            cardEl?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      });
    });
  };

  // SSE connection effect - connect when signed in and has active trades
  createEffect(() => {
    const hasActiveTrades = slots().some(s => s.isOrdered);

    if (isSignedIn() && hasActiveTrades) {
      connectSSE();
    } else {
      disconnectSSE();
    }

    onCleanup(() => {
      disconnectSSE();
    });
  });

  // Page visibility handling - disconnect SSE when tab is hidden, reconnect when visible
  createEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        disconnectSSE();
      } else if (isSignedIn() && slots().some(s => s.isOrdered)) {
        connectSSE();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    onCleanup(() => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    });
  });

  const handlePlaceOrder = (slotIndex: number) => {
    if (!isSignedIn()) {
      setPendingOrderSlot(slotIndex);
      setShowSignInModal(true);
      return;
    }
    placeOrder(slotIndex);
  };

  const placeOrder = (slotIndex: number) => {
    const recs = recommendations();
    const currentRecIndex = recIndex();
    const currentActiveCount = activeOrderCount();

    // Mark current slot as ordered
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex || !slot.recommendation) return slot;
      return {
        ...slot,
        isOrdered: true,
        phase: 'buying' as const,
        quantity: slot.recommendation.quantity,
        filled: 0,
        sold: 0
      };
    }));

    // If we haven't hit the limit, add a recommendation to the next slot
    if (currentActiveCount + 1 < maxSlots() && currentRecIndex < recs.length) {
      const nextSlotIndex = currentActiveCount + 1; // Next slot after this order
      setSlots(prev => prev.map((slot, i) => {
        if (i !== nextSlotIndex) return slot;
        return {
          ...slot,
          recommendation: recs[currentRecIndex] || null
        };
      }));
      setRecIndex(currentRecIndex + 1);
    }
  };

  const handleSignIn = () => {
    // Redirect to login page - the user will be redirected back after auth
    window.location.href = '/login';
  };

  // Index tracking which recommendation to show next from the cached list
  const [browseIndex, setBrowseIndex] = createSignal(0);
  const [fetchingMore, setFetchingMore] = createSignal(false);

  const skipRecommendation = async (slotIndex: number) => {
    const currentSlot = slots()[slotIndex];
    if (!currentSlot.recommendation || fetchingMore()) return;

    const recs = recommendations();
    const currentBrowseIndex = browseIndex();

    // Find next recommendation that isn't already in an ordered slot
    const orderedItemIds = new Set(
      slots().filter(s => s.isOrdered && s.recommendation).map(s => s.recommendation!.itemId)
    );

    let nextRec: Recommendation | null = null;
    let nextIndex = currentBrowseIndex;

    // Search forward through recommendations for one not already ordered
    for (let i = currentBrowseIndex; i < recs.length; i++) {
      if (!orderedItemIds.has(recs[i].itemId) && recs[i].itemId !== currentSlot.recommendation.itemId) {
        nextRec = recs[i];
        nextIndex = i + 1;
        break;
      }
    }

    if (nextRec) {
      // Save current to history for "back" button
      setSkippedHistory(prev => [...prev, currentSlot.recommendation!]);
      setBrowseIndex(nextIndex);

      setSlots(prev => prev.map((slot, i) => {
        if (i !== slotIndex) return slot;
        return { ...slot, recommendation: nextRec };
      }));
    } else {
      // No more in cache - fetch more from API with excludes
      setFetchingMore(true);

      // Build exclude list: ordered items + skipped history + current
      const excludeIds = [
        ...Array.from(orderedItemIds),
        ...skippedHistory().map(r => r.itemId),
        currentSlot.recommendation.itemId
      ];

      try {
        // Use slot's settings for the request
        const slotSettings = currentSlot.settings;
        const params = new URLSearchParams({
          capital: userCapital().toString(),
          style: slotSettings.style,
          risk: slotSettings.risk,
          margin: slotSettings.margin
        });
        if (excludeIds.length > 0) {
          params.append('exclude', excludeIds.join(','));
        }

        const response = await fetch(`/api/recommendations?${params}`);
        const data = await response.json();

        if (data.success && data.data?.length > 0) {
          // Add new recs to cache
          setRecommendations(prev => [...prev, ...data.data]);
          // Save current to history
          setSkippedHistory(prev => [...prev, currentSlot.recommendation!]);
          // Show first new rec
          setSlots(prev => prev.map((slot, i) => {
            if (i !== slotIndex) return slot;
            return { ...slot, recommendation: data.data[0] };
          }));
          setBrowseIndex(recs.length + 1);
        } else {
          addToast('No more recommendations available', 'info');
        }
      } catch (err) {
        console.error('Failed to fetch more recommendations:', err);
        addToast('Failed to load more recommendations', 'error');
      } finally {
        setFetchingMore(false);
      }
    }
  };

  const goBackRecommendation = (slotIndex: number) => {
    const history = skippedHistory();
    if (history.length === 0) return;

    const previousRec = history[history.length - 1];

    // Decrement browse index
    setBrowseIndex(i => Math.max(0, i - 1));

    // Pop from history and set as current
    setSkippedHistory(prev => prev.slice(0, -1));
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return {
        ...slot,
        recommendation: previousRec
      };
    }));
  };

  // Item search
  let searchTimeout: ReturnType<typeof setTimeout> | null = null;

  const handleSearchInput = (value: string) => {
    setSearchQuery(value);
    if (searchTimeout) clearTimeout(searchTimeout);

    if (value.length < 2) {
      setSearchResults([]);
      return;
    }

    setSearchLoading(true);
    searchTimeout = setTimeout(async () => {
      try {
        const res = await fetch(`/api/items/search?q=${encodeURIComponent(value)}`);
        if (res.ok) {
          const data = await res.json();
          setSearchResults(data.data || []);
        }
      } catch {
        setSearchResults([]);
      } finally {
        setSearchLoading(false);
      }
    }, 300);
  };

  const selectSearchItem = async (item: { id: number; name: string }) => {
    const slotIndex = searchSlot();
    if (slotIndex === null) return;

    // Fetch item details and create a custom recommendation
    try {
      const res = await fetch(`/api/items/${item.id}`);
      if (res.ok) {
        const itemData = await res.json();
        const customRec: Recommendation = {
          id: `custom-${item.id}-${Date.now()}`,
          itemId: item.id,
          item: item.name,
          buyPrice: itemData.buyPrice || itemData.price || 1000,
          sellPrice: itemData.sellPrice || Math.floor((itemData.price || 1000) * 1.05),
          quantity: itemData.suggestedQuantity || 100,
          expectedProfit: itemData.expectedProfit || 0,
          capitalRequired: itemData.capitalRequired || 0,
          confidence: 0.5,
          fillProbability: 0.5,
          fillConfidence: 0.5,
          expectedHours: 4,
          trend: 'stable' as const,
          reason: 'Custom item selected by user',
          priceHistory: itemData.priceHistory || [],
          volume24h: itemData.volume24h || 0,
          modelId: 'custom'
        };

        setSlots(prev => prev.map((slot, i) => {
          if (i !== slotIndex) return slot;
          return {
            ...slot,
            recommendation: customRec,
            isCustomItem: true,
            originalRecommendation: slot.isCustomItem ? slot.originalRecommendation : slot.recommendation
          };
        }));
      }
    } catch {
      // If API fails, create a basic recommendation
      const basicRec: Recommendation = {
        id: `custom-${item.id}-${Date.now()}`,
        itemId: item.id,
        item: item.name,
        buyPrice: 1000,
        sellPrice: 1050,
        quantity: 100,
        expectedProfit: 5000,
        capitalRequired: 100000,
        confidence: 0.5,
        fillProbability: 0.5,
        fillConfidence: 0.5,
        expectedHours: 4,
        trend: 'stable' as const,
        reason: 'Custom item selected by user',
        priceHistory: [],
        volume24h: 0,
        modelId: 'custom'
      };

      setSlots(prev => prev.map((slot, i) => {
        if (i !== slotIndex) return slot;
        return {
          ...slot,
          recommendation: basicRec,
          isCustomItem: true,
          originalRecommendation: slot.isCustomItem ? slot.originalRecommendation : slot.recommendation
        };
      }));
    }

    // Close modal
    setSearchSlot(null);
    setSearchQuery('');
    setSearchResults([]);
  };

  const restoreOriginalRecommendation = (slotIndex: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex || !slot.originalRecommendation) return slot;
      return {
        ...slot,
        recommendation: slot.originalRecommendation,
        isCustomItem: false,
        originalRecommendation: null
      };
    }));
  };

  const updateQuantity = (slotIndex: number, quantity: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return { ...slot, quantity: Math.max(0, quantity) };
    }));
  };

  const updateFilled = (slotIndex: number, filled: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return { ...slot, filled: Math.max(0, Math.min(slot.quantity, filled)) };
    }));
  };

  const updateSold = (slotIndex: number, sold: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return { ...slot, sold: Math.max(0, Math.min(slot.filled, sold)) };
    }));
  };

  const incrementSold = (slotIndex: number, amount: number) => {
    const slot = slots()[slotIndex];
    const newSold = Math.max(0, Math.min(slot.filled, slot.sold + amount));
    updateSold(slotIndex, newSold);
  };

  const startSelling = (slotIndex: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return {
        ...slot,
        phase: 'selling' as const,
        sold: 0
      };
    }));
  };

  const completeOrder = async (slotIndex: number) => {
    const slot = slots()[slotIndex];
    if (!slot.recommendation) return;

    // Report the trade
    try {
      const profit = (slot.recommendation.sellPrice - slot.recommendation.buyPrice) * slot.sold;
      await fetch('/api/trades/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemId: slot.recommendation.itemId,
          itemName: slot.recommendation.item,
          buyPrice: slot.recommendation.buyPrice,
          sellPrice: slot.recommendation.sellPrice,
          quantity: slot.sold,
          profit
        })
      });
    } catch {
      // Continue even if report fails
    }

    // Use same consolidation logic as cancel
    const currentSlots = slots();
    const recs = recommendations();

    // Collect remaining active orders (excluding the one being completed)
    const remainingOrders = currentSlots
      .filter((s, i) => s.isOrdered && i !== slotIndex)
      .map(s => ({ ...s }));

    // Get a recommendation for the "next" slot
    let nextRec = recs[recIndex()] || null;
    if (nextRec) {
      setRecIndex(i => i + 1);
    } else {
      // If no more recommendations in queue, use the completed order's recommendation
      // This ensures we always have at least one card showing
      nextRec = slot.recommendation;
    }

    // Rebuild slots: active orders first, then one recommendation, then empty
    setSlots(prev => prev.map((s, i) => {
      if (i < remainingOrders.length) {
        return remainingOrders[i];
      } else if (i === remainingOrders.length) {
        return {
          ...s,
          isOrdered: false,
          phase: 'buying' as const,
          quantity: 0,
          filled: 0,
          sold: 0,
          recommendation: nextRec,
          showSettings: false,
          pendingUpdate: null,
          lastCheckedAt: null
        };
      } else {
        return {
          ...s,
          isOrdered: false,
          phase: 'buying' as const,
          quantity: 0,
          filled: 0,
          sold: 0,
          recommendation: null,
          showSettings: false,
          pendingUpdate: null,
          lastCheckedAt: null
        };
      }
    }));
  };

  const updateSlotSettings = <K extends keyof SlotSettings>(slotIndex: number, key: K, value: SlotSettings[K]) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return {
        ...slot,
        settings: { ...slot.settings, [key]: value }
      };
    }));
  };

  const toggleSettings = (slotIndex: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      const opening = !slot.showSettings;
      return {
        ...slot,
        showSettings: opening,
        // Save original settings when opening, clear when closing
        originalSettings: opening ? { ...slot.settings } : null
      };
    }));
  };

  // Confirm slot settings and fetch new recommendation with those settings
  const confirmSlotSettings = async (slotIndex: number) => {
    const slot = slots()[slotIndex];
    if (!slot) return;

    // Close settings panel and clear original
    setSlots(prev => prev.map((s, i) => {
      if (i !== slotIndex) return s;
      return { ...s, showSettings: false, originalSettings: null };
    }));

    // Fetch new recommendation with slot's settings
    const params = new URLSearchParams({
      slots: '1',
      capital: userCapital().toString(),
      style: slot.settings.style,
      risk: slot.settings.risk,
      margin: slot.settings.margin
    });

    // Exclude current item and active trades
    const excludeIds = slots()
      .filter(s => s.isOrdered && s.recommendation)
      .map(s => s.recommendation!.itemId);
    if (slot.recommendation) {
      excludeIds.push(slot.recommendation.itemId);
    }
    if (excludeIds.length > 0) {
      params.append('exclude', excludeIds.join(','));
    }
    // Bypass cache after per-card settings change
    params.append('fresh', '1');

    try {
      const response = await fetch(`/api/recommendations?${params}`);
      const data = await response.json();

      if (data.success && data.data?.length > 0) {
        setSlots(prev => prev.map((s, i) => {
          if (i !== slotIndex) return s;
          return { ...s, recommendation: data.data[0], isCustomItem: false };
        }));
      } else {
        addToast('No recommendations for these settings', 'info');
      }
    } catch (err) {
      console.error('Failed to fetch recommendation with new settings:', err);
      addToast('Failed to update recommendation', 'error');
    }
  };

  const cancelOrder = (slotIndex: number) => {
    const currentSlots = slots();
    const canceledSlot = currentSlots[slotIndex];

    // Collect remaining active orders (excluding the one being canceled)
    const remainingOrders = currentSlots
      .filter((s, i) => s.isOrdered && i !== slotIndex)
      .map(s => ({ ...s }));

    // Use the recommendation from the canceled order (go back to same rec)
    const recToRestore = canceledSlot.recommendation;

    // Rebuild slots: active orders first, then one recommendation (the canceled one), then empty
    setSlots(prev => prev.map((slot, i) => {
      if (i < remainingOrders.length) {
        // Active order slot
        return remainingOrders[i];
      } else if (i === remainingOrders.length && recToRestore) {
        // Recommendation slot - restore the canceled order's recommendation
        return {
          ...slot,
          isOrdered: false,
          phase: 'buying' as const,
          quantity: 0,
          filled: 0,
          sold: 0,
          recommendation: recToRestore,
          showSettings: false,
          pendingUpdate: null,
          lastCheckedAt: null
        };
      } else {
        // Empty/locked slot
        return {
          ...slot,
          isOrdered: false,
          phase: 'buying' as const,
          quantity: 0,
          filled: 0,
          sold: 0,
          recommendation: null,
          showSettings: false,
          pendingUpdate: null,
          lastCheckedAt: null
        };
      }
    }));
  };

  const goBackToBuying = (slotIndex: number) => {
    setSlots(prev => prev.map((slot, i) => {
      if (i !== slotIndex) return slot;
      return {
        ...slot,
        phase: 'buying' as const,
        sold: 0
      };
    }));
  };

  // Accept an update recommendation - stages it for final confirmation
  const acceptUpdate = async (slotIndex: number) => {
    const slot = slots()[slotIndex];
    const update = slot.pendingUpdate;
    if (!update || !slot.recommendation) return;

    try {
      // Report acceptance to backend
      await fetch(`/api/trades/updates/${update.id}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'accept' })
      });
    } catch {
      // Continue even if report fails
    }

    // Stage the update for final confirmation (shows yellow highlights)
    setSlots(prev => prev.map((s, i) => {
      if (i !== slotIndex) return s;
      return {
        ...s,
        pendingUpdate: null,
        acceptedUpdate: update
      };
    }));

    addToast({
      type: 'info',
      title: 'Update Staged',
      message: 'Review the changes and click Update to confirm.',
      duration: 3000
    });
  };

  // Confirm and apply the accepted update
  const confirmUpdate = (slotIndex: number) => {
    const slot = slots()[slotIndex];
    const update = slot.acceptedUpdate;
    if (!update || !slot.recommendation) return;

    // Apply the update based on type
    switch (update.type) {
      case 'SWITCH_ITEM':
        if (update.newItem) {
          setSlots(prev => prev.map((s, i) => {
            if (i !== slotIndex) return s;
            return {
              ...s,
              phase: 'buying' as const,
              quantity: update.newItem!.quantity,
              filled: 0,
              sold: 0,
              recommendation: {
                ...s.recommendation!,
                item: update.newItem!.item,
                itemId: update.newItem!.itemId,
                buyPrice: update.newItem!.buyPrice,
                sellPrice: update.newItem!.sellPrice,
                quantity: update.newItem!.quantity,
                expectedProfit: update.newItem!.expectedProfit,
                confidence: update.newItem!.confidence
              },
              acceptedUpdate: null
            };
          }));
        }
        break;

      case 'SELL_NOW':
        setSlots(prev => prev.map((s, i) => {
          if (i !== slotIndex) return s;
          return {
            ...s,
            phase: 'selling' as const,
            sold: 0,
            recommendation: update.adjustedSellPrice
              ? { ...s.recommendation!, sellPrice: update.adjustedSellPrice }
              : s.recommendation,
            acceptedUpdate: null
          };
        }));
        break;

      case 'ADJUST_PRICE':
        if (update.newSellPrice) {
          setSlots(prev => prev.map((s, i) => {
            if (i !== slotIndex) return s;
            return {
              ...s,
              recommendation: {
                ...s.recommendation!,
                sellPrice: update.newSellPrice!
              },
              acceptedUpdate: null
            };
          }));
        }
        break;
    }

    addToast({
      type: 'success',
      title: 'Update Applied',
      message: `${slot.recommendation.item}: ${update.type.replace('_', ' ').toLowerCase()} accepted`,
      duration: 3000
    });
  };

  // Ignore an update recommendation (keeps showing until conditions change)
  const ignoreUpdate = async (slotIndex: number) => {
    const slot = slots()[slotIndex];
    const update = slot.pendingUpdate;
    if (!update) return;

    try {
      await fetch(`/api/trades/updates/${update.id}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'ignore' })
      });
    } catch {
      // Silently fail
    }

    // Clear the pending update
    setSlots(prev => prev.map((s, i) => {
      if (i !== slotIndex) return s;
      return {
        ...s,
        pendingUpdate: null
      };
    }));
  };

  const trendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return '↑';
      case 'down': return '↓';
      default: return '→';
    }
  };

  const trendClass = (trend: string) => {
    switch (trend) {
      case 'up': return 'trend-up';
      case 'down': return 'trend-down';
      default: return 'trend-stable';
    }
  };

  // Fallback for broken item images
  const handleImageError = (e: Event) => {
    const img = e.target as HTMLImageElement;
    // Use a simple gray placeholder SVG
    img.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"%3E%3Crect fill="%23383838" width="32" height="32" rx="4"/%3E%3Ctext x="16" y="20" text-anchor="middle" fill="%23707070" font-size="10"%3E?%3C/text%3E%3C/svg%3E';
    img.onerror = null; // Prevent infinite loop
  };

  const generateSparkline = (prices: number[] | undefined, _trend: string, style: 'passive' | 'hybrid' | 'active', slotIndex: number) => {
    if (!prices || prices.length < 2) return null;

    // Time range based on style (x2 the flip window)
    // Active: 4-8hr flip → 16hrs, Hybrid: 12-24hr → 48hrs, Passive: 48hr → 96hrs
    const hoursToShow = style === 'active' ? 16 : style === 'hybrid' ? 48 : 96;
    const pointsToShow = Math.min(hoursToShow, prices.length);
    const displayPrices = prices.slice(-pointsToShow);

    const width = 200;
    const height = 50;
    const padding = 4;

    const min = Math.min(...displayPrices);
    const max = Math.max(...displayPrices);
    const range = max - min || 1;

    const points = displayPrices.map((price, i) => {
      const x = padding + (i / (displayPrices.length - 1)) * (width - padding * 2);
      const y = height - padding - ((price - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    });

    const linePath = `M ${points.join(' L ')}`;
    const areaPath = `${linePath} L ${width - padding},${height - padding} L ${padding},${height - padding} Z`;

    // Calculate actual trend from price data (compare first vs last in display window)
    const startPrice = displayPrices[0];
    const endPrice = displayPrices[displayPrices.length - 1];
    const priceChange = endPrice - startPrice;
    const changePercent = Math.abs(priceChange / startPrice);
    // Only show up/down if change is > 0.5%, otherwise stable
    const actualTrend = changePercent < 0.005 ? 'stable' : priceChange > 0 ? 'up' : 'down';

    const strokeColor = actualTrend === 'up' ? 'var(--success)' : actualTrend === 'down' ? 'var(--danger)' : 'var(--text-muted)';
    const fillColor = actualTrend === 'up' ? 'var(--success-light)' : actualTrend === 'down' ? 'var(--danger-light)' : 'var(--bg-tertiary)';

    const timeLabel = `${hoursToShow}h`;

    // Position label where there's space - check end of line position
    const lastPrice = displayPrices[displayPrices.length - 1];
    const lastY = height - padding - ((lastPrice - min) / range) * (height - padding * 2);
    const labelAtBottom = lastY < height / 2; // Line ends high, put label at bottom
    const labelY = labelAtBottom ? height - padding - 2 : padding + 10;

    const calcHoverData = (clientX: number, rect: DOMRect) => {
      const xPercent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const dataIndex = Math.round(xPercent * (displayPrices.length - 1));
      const clampedIndex = Math.max(0, Math.min(displayPrices.length - 1, dataIndex));
      const price = displayPrices[clampedIndex];
      const hoursAgo = displayPrices.length - 1 - clampedIndex;
      setHoverData({ slotIndex, x: xPercent * 100, price, hoursAgo });
    };

    const handleMouseDown = (e: MouseEvent) => {
      e.preventDefault();
      const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
      calcHoverData(e.clientX, rect);

      const onMove = (moveE: MouseEvent) => {
        moveE.preventDefault();
        calcHoverData(moveE.clientX, rect);
      };

      const onUp = () => {
        setHoverData(null);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };

      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    };

    const hover = hoverData();
    const showHover = hover && hover.slotIndex === slotIndex;
    const hoverX = showHover ? (padding + (hover.x / 100) * (width - padding * 2)) : 0;
    const hoverPrice = showHover ? hover.price : 0;
    const hoverY = showHover ? (height - padding - ((hoverPrice - min) / range) * (height - padding * 2)) : 0;
    // Clamp tooltip position so it stays within bounds
    const tooltipPercent = showHover ? Math.max(15, Math.min(85, hover.x)) : 0;

    return (
      <div class="sparkline-wrapper">
        {showHover && (
          <div class="sparkline-tooltip" style={{ left: `${tooltipPercent}%` }}>
            {formatGold(hoverPrice, 'gp')}
          </div>
        )}
        <svg
          class="sparkline"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          onMouseDown={handleMouseDown}
        >
          <path d={areaPath} fill={fillColor} opacity="0.3" />
          <path d={linePath} fill="none" stroke={strokeColor} stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          <text x={width - padding - 2} y={labelY} text-anchor="end" fill="var(--text-muted)" font-size="9">{timeLabel}</text>
          {showHover && (
            <>
              <line x1={hoverX} y1={padding} x2={hoverX} y2={height - padding} stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="2,2" />
              <circle cx={hoverX} cy={hoverY} r="4" fill={strokeColor} />
            </>
          )}
        </svg>
      </div>
    );
  };

  return (
    <div class="order-grid-container">
      <Show when={loading()}>
        <div class="grid-loading">
          <div class="spinner"></div>
          <p class="text-secondary mt-3">Loading recommendations...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="grid-error">
          <p class="text-danger mb-3">{error()}</p>
          <button class="btn btn-primary" onClick={fetchRecommendations}>
            Try Again
          </button>
        </div>
      </Show>

      <Show when={!loading() && !error()}>
        <div class="slots-header">
          {/* Stats dropdown - left */}
          <div class="header-dropdown-wrap">
            <button
              class={`header-dropdown-btn ${showHistoryPanel() ? 'active' : ''}`}
              onClick={() => { setShowHistoryPanel(!showHistoryPanel()); setShowStatsPanel(false); }}
            >
              <svg class="dropdown-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 3v18h18"/>
                <path d="M18 9l-5 5-4-4-3 3"/>
              </svg>
              <span>Stats</span>
              <svg class="dropdown-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M6 9l6 6 6-6"/>
              </svg>
            </button>
            <Show when={showHistoryPanel()}>
              <div class="header-panel left stats-dashboard">
                <div class="stats-dashboard-grid">
                  <div class="stats-dashboard-item">
                    <span class="stats-dashboard-label">Invested</span>
                    <span class="stats-dashboard-value">{formatGold(tradeStats().totalCapital)}</span>
                  </div>
                  <div class="stats-dashboard-item">
                    <span class="stats-dashboard-label">Expected</span>
                    <span class="stats-dashboard-value profit">+{formatGold(tradeStats().expectedProfit)}</span>
                  </div>
                  <div class="stats-dashboard-item">
                    <span class="stats-dashboard-label">Buying</span>
                    <span class="stats-dashboard-value">{tradeStats().buyingCount}</span>
                  </div>
                  <div class="stats-dashboard-item">
                    <span class="stats-dashboard-label">Selling</span>
                    <span class="stats-dashboard-value">{tradeStats().sellingCount}</span>
                  </div>
                </div>
                <a href="/stats" class="stats-details-link" onClick={() => setShowHistoryPanel(false)}>
                  View Details
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M5 12h14M12 5l7 7-7 7"/>
                  </svg>
                </a>
              </div>
            </Show>
          </div>

          {/* Trades count - center */}
          <span class="slots-info-center">
            {activeOrderCount()} of {maxSlots()} trades active
          </span>

          {/* Capital dropdown - right */}
          <div class="capital-indicator-wrap">
            <button
              class={`capital-indicator ${showStatsPanel() ? 'active' : ''}`}
              onClick={() => { setShowStatsPanel(!showStatsPanel()); setShowHistoryPanel(false); }}
            >
              <svg class="capital-icon" viewBox="0 0 24 24" fill="none">
                <ellipse cx="12" cy="18" rx="8" ry="3" fill="#b8860b"/>
                <ellipse cx="12" cy="17" rx="8" ry="3" fill="#daa520"/>
                <ellipse cx="12" cy="14" rx="8" ry="3" fill="#b8860b"/>
                <ellipse cx="12" cy="13" rx="8" ry="3" fill="#daa520"/>
                <ellipse cx="12" cy="10" rx="8" ry="3" fill="#b8860b"/>
                <ellipse cx="12" cy="9" rx="8" ry="3" fill="#ffd700"/>
              </svg>
              <span class="capital-amount">{formatGold(userCapital())}</span>
              <svg class="capital-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M6 9l6 6 6-6"/>
              </svg>
            </button>
            <Show when={showStatsPanel()}>
              <div class="stats-panel">
                <div class="stats-panel-header">
                  <span>Settings</span>
                  <div class="stats-header-actions">
                    <Show when={hasPendingChanges()} fallback={
                      <button class="stats-close" aria-label="Close settings" onClick={() => setShowStatsPanel(false)}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                          <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                      </button>
                    }>
                      <button class="stats-action discard" aria-label="Discard changes" onClick={() => { discardSettingsChanges(); setShowStatsPanel(false); }}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                          <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                      </button>
                      <button class="stats-action confirm" aria-label="Confirm changes" onClick={confirmSettingsChanges}>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                          <path d="M20 6L9 17l-5-5"/>
                        </svg>
                      </button>
                    </Show>
                  </div>
                </div>
                <div class="stats-panel-content">
                  {/* Capital Section */}
                  <div class="panel-section">
                    <label class="panel-label">Total Capital</label>
                    <div class="capital-input-row">
                      <input
                        type="text"
                        class="capital-input"
                        value={formatGold(userCapital())}
                        onFocus={(e) => e.currentTarget.value = userCapital().toString()}
                        onBlur={(e) => {
                          const val = parseInt(e.currentTarget.value.replace(/[^0-9]/g, ''), 10);
                          if (val > 0) updateCapital(val);
                          e.currentTarget.value = formatGold(userCapital());
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.currentTarget.blur();
                          }
                        }}
                      />
                      <span class="capital-hint">Click to edit</span>
                    </div>
                  </div>

                  {/* Default Settings Section */}
                  <div class="panel-section">
                    <label class="panel-label">Default Flip Settings</label>

                    <div class="setting-row">
                      <span class="setting-label">Style</span>
                      <div class="setting-btns">
                        <button class={`setting-btn ${globalDefaults().style === 'passive' ? 'active' : ''}`} onClick={() => updateGlobalDefault('style', 'passive')}>Passive</button>
                        <button class={`setting-btn ${globalDefaults().style === 'hybrid' ? 'active' : ''}`} onClick={() => updateGlobalDefault('style', 'hybrid')}>Hybrid</button>
                        <button class={`setting-btn ${globalDefaults().style === 'active' ? 'active' : ''}`} onClick={() => updateGlobalDefault('style', 'active')}>Active</button>
                      </div>
                    </div>

                    <div class="setting-row">
                      <span class="setting-label">Risk</span>
                      <div class="setting-btns">
                        <button class={`setting-btn ${globalDefaults().risk === 'low' ? 'active' : ''}`} onClick={() => updateGlobalDefault('risk', 'low')}>Low</button>
                        <button class={`setting-btn ${globalDefaults().risk === 'medium' ? 'active' : ''}`} onClick={() => updateGlobalDefault('risk', 'medium')}>Med</button>
                        <button class={`setting-btn ${globalDefaults().risk === 'high' ? 'active' : ''}`} onClick={() => updateGlobalDefault('risk', 'high')}>High</button>
                      </div>
                    </div>

                    <div class="setting-row">
                      <span class="setting-label">Margin</span>
                      <div class="setting-btns">
                        <button class={`setting-btn ${globalDefaults().margin === 'conservative' ? 'active' : ''}`} onClick={() => updateGlobalDefault('margin', 'conservative')}>Safe</button>
                        <button class={`setting-btn ${globalDefaults().margin === 'moderate' ? 'active' : ''}`} onClick={() => updateGlobalDefault('margin', 'moderate')}>Mid</button>
                        <button class={`setting-btn ${globalDefaults().margin === 'aggressive' ? 'active' : ''}`} onClick={() => updateGlobalDefault('margin', 'aggressive')}>Aggro</button>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            </Show>
          </div>
        </div>
        <div class="order-grid">
          <For each={slots()}>
            {(slot, index) => (
              <div class={`order-card ${slot.isOrdered ? 'is-ordered' : ''} ${slot.recommendation ? 'has-rec' : ''} ${index() >= visibleSlots() ? 'is-locked' : ''} ${index() >= maxSlots() ? 'is-pro-locked' : ''} ${!slot.recommendation && index() < visibleSlots() ? 'is-empty' : ''}`}>
                {/* Pro Locked Slot - Beyond tier limit */}
                <Show when={index() >= maxSlots()}>
                  <div class="card-content pro-locked-slot">
                    <div class="pro-lock-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                      </svg>
                    </div>
                    <span class="pro-locked-text">Upgrade to Pro</span>
                    <span class="pro-locked-subtext">Unlock {MAX_SLOTS - TIER_LIMITS.free} more slots</span>
                    <a href="/account" class="btn btn-upgrade btn-sm">Upgrade</a>
                  </div>
                </Show>

                {/* Locked Slot - Within tier but needs order */}
                <Show when={index() >= visibleSlots() && index() < maxSlots()}>
                  <div class="card-content locked-slot">
                    <Tooltip text="Start a trade in an earlier slot to unlock this one. Slots unlock sequentially." position="bottom">
                      <div class="locked-content">
                        <div class="lock-icon">
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
                            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                          </svg>
                        </div>
                        <span class="locked-text">Place an order to unlock</span>
                        <span class="locked-subtext">Slot {index() + 1}</span>
                      </div>
                    </Tooltip>
                  </div>
                </Show>

                {/* Always render the GePT badge outside conditional views - using index only to prevent re-render */}
                {/* Persistent badge - always in DOM, visibility via CSS */}
                <div class={`persistent-badge ${index() < visibleSlots() && slot.recommendation && !slot.isCustomItem && !slot.isOrdered && !slot.showSettings && searchSlot() !== index() ? '' : 'hidden'}`}>
                  <Tooltip text="AI-selected flip based on your settings and current market conditions" position="bottom">
                    <span class="ai-badge-wrapper">
                      <span class="ai-badge">
                        <span class="ai-badge-inner">
                          <span class="ai-sword-bg" aria-label="AI Rec"></span>
                          <span class="ai-badge-text">GePT</span>
                        </span>
                      </span>
                      <span class="ai-badge-label">Recommended</span>
                    </span>
                  </Tooltip>
                </div>

                <Show when={index() < visibleSlots() && slot.recommendation && !slot.isOrdered && !slot.showSettings}>
                    <div class="card-content recommendation-view">
                      <div class="card-header">
                        {/* Normal header content - use CSS hide instead of Show to prevent re-render */}
                        <div class={`item-icon ${searchSlot() === index() ? 'hidden' : ''}`}>
                          <img
                            src={itemImageUrl(slot.recommendation!.itemId)}
                            alt={slot.recommendation!.item}
                            loading="lazy"
                            onError={handleImageError}
                          />
                        </div>
                        <span class={`item-name ${searchSlot() === index() ? 'hidden' : ''}`}>{slot.recommendation!.item}</span>
                        <span class={`custom-badge ${searchSlot() === index() || !slot.isCustomItem ? 'hidden' : ''}`}>Custom</span>
                        {/* Badge spacer - actual badge is rendered outside this Show for persistence */}
                        <span class={`badge-spacer ${searchSlot() === index() || slot.isCustomItem ? 'hidden' : ''}`}></span>

                        {/* Search input - expands from search button */}
                        <Show when={searchSlot() === index()}>
                          <div class="header-search">
                            <svg class="header-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <circle cx="11" cy="11" r="8"/>
                              <path d="M21 21l-4.35-4.35"/>
                            </svg>
                            <input
                              type="text"
                              class="header-search-input"
                              placeholder="Search items..."
                              value={searchQuery()}
                              onInput={(e) => handleSearchInput(e.currentTarget.value)}
                              autofocus
                            />
                          </div>
                        </Show>

                        {/* Search/Close button - stays in same position */}
                        <Show when={searchSlot() !== index()}>
                          <button class="gear-btn" title="Search Items" aria-label="Search items" onClick={() => setSearchSlot(index())}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                              <circle cx="11" cy="11" r="8"/>
                              <path d="M21 21l-4.35-4.35"/>
                            </svg>
                          </button>
                        </Show>
                        <Show when={searchSlot() === index()}>
                          <button class="gear-btn" title="Close Search" aria-label="Close search" onClick={() => { setSearchSlot(null); setSearchQuery(''); setSearchResults([]); }}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                              <path d="M18 6L6 18M6 6l12 12"/>
                            </svg>
                          </button>
                        </Show>

                        <button class="gear-btn" title="Slot Settings" aria-label="Slot settings" onClick={() => toggleSettings(index())}>
                          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                            <circle cx="12" cy="12" r="3"/>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                          </svg>
                        </button>
                      </div>

                      {/* Search results - replaces card content when searching */}
                      <Show when={searchSlot() === index()}>
                        <div class="search-results-area">
                          <Show when={searchLoading()}>
                            <div class="search-loading">
                              <div class="spinner"></div>
                            </div>
                          </Show>
                          <Show when={!searchLoading() && searchResults().length > 0}>
                            <div class="search-results">
                              <For each={searchResults()}>
                                {(item) => (
                                  <button class="search-result" onClick={() => selectSearchItem(item)}>
                                    <img src={itemImageUrl(item.id)} alt={item.name} loading="lazy" onError={handleImageError} />
                                    <span>{item.name}</span>
                                  </button>
                                )}
                              </For>
                            </div>
                          </Show>
                          <Show when={!searchLoading() && searchQuery().length > 2 && searchResults().length === 0}>
                            <p class="search-empty">No items found</p>
                          </Show>
                          <Show when={!searchLoading() && searchQuery().length <= 2 && searchResults().length === 0}>
                            <p class="search-hint">Type to search for items...</p>
                          </Show>
                        </div>
                      </Show>

                      {/* Normal card content - hidden during search */}
                      <Show when={searchSlot() !== index()}>
                      <div class="price-row">
                        <div class="price-item">
                          <span class="price-label">Buy</span>
                          <span class="price-value">{formatGold(slot.recommendation!.buyPrice, 'gp')}</span>
                        </div>
                        <span class="price-arrow">→</span>
                        <div class="price-item">
                          <span class="price-label">Sell</span>
                          <span class="price-value">{formatGold(slot.recommendation!.sellPrice, 'gp')}</span>
                        </div>
                      </div>

                      <div class="stats-row compact">
                        <div class="stat-item">
                          <span class="stat-value">{slot.recommendation!.quantity.toLocaleString()}</span>
                          <span class="stat-label">Qty</span>
                        </div>
                        <div class="stat-item">
                          <span class="stat-value">{formatGold(slot.recommendation!.capitalRequired)}</span>
                          <span class="stat-label">Total</span>
                        </div>
                        <div class="stat-item profit">
                          <span class="stat-value">+{formatGold(slot.recommendation!.expectedProfit)}</span>
                          <span class="stat-label">Profit</span>
                        </div>
                      </div>

                      <div class="flex-spacer"></div>

                      <Show when={showInfoSlot() !== index()}>
                        <div class="sparkline-container">
                          {generateSparkline(slot.recommendation!.priceHistory, slot.recommendation!.trend, slot.settings.style, index())}
                          <Show when={slot.isCustomItem && (!slot.recommendation!.priceHistory || slot.recommendation!.priceHistory.length < 2)}>
                            <div class="sparkline-placeholder">
                              <span>No price history</span>
                            </div>
                          </Show>
                        </div>
                      </Show>

                      <Show when={showInfoSlot() === index()}>
                        <div class="info-panel">
                          <div class="info-row">
                            <Tooltip text="Estimated time for orders to fill based on market activity" position="left">
                              <span class="info-label info-label-tooltip">Time to Fill <InfoIcon /></span>
                            </Tooltip>
                            <span class="info-value">{slot.recommendation!.expectedHours}h</span>
                          </div>
                          <div class="info-row">
                            <Tooltip text="Model confidence in this recommendation. Higher = more certain." position="left">
                              <span class="info-label info-label-tooltip">Confidence <InfoIcon /></span>
                            </Tooltip>
                            <span class="info-value">{formatPercent(slot.recommendation!.confidence)}</span>
                          </div>
                          <div class="info-row">
                            <span class="info-label">Volume 24h</span>
                            <span class="info-value">{formatRsNumber(slot.recommendation!.volume24h, 3)}</span>
                          </div>
                          <div class="info-row">
                            <span class="info-label">Live Price</span>
                            <span class="info-value">{formatRsPrice(slot.recommendation!.buyPrice)}</span>
                          </div>
                        </div>
                      </Show>

                      <div class="action-row rec-actions">
                        <button
                          class={`btn btn-ghost btn-sm btn-info ${showInfoSlot() === index() ? 'active' : ''}`}
                          title="More details"
                          onClick={() => setShowInfoSlot(showInfoSlot() === index() ? null : index())}
                        >
                          <svg class="info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="16" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12.01" y2="8"/>
                          </svg>
                        </button>
                        <Show when={slot.isCustomItem && slot.originalRecommendation}>
                          <button
                            class="ai-rec-btn"
                            onClick={() => restoreOriginalRecommendation(index())}
                            title="Back to AI recommendation"
                          >
                            <span class="ai-badge-wrap">
                              <span class="ai-badge">
                                <span class="ai-badge-inner">
                                  <img src="/images/logo-icon.png" alt="AI Rec" class="ai-sword" />
                                  <span class="ai-badge-text">GePT</span>
                                </span>
                              </span>
                              <span class="ai-badge-label">Recommended</span>
                            </span>
                          </button>
                        </Show>
                        <Show when={!slot.isCustomItem}>
                          <button
                            class="btn btn-ghost btn-sm btn-nav"
                            onClick={() => goBackRecommendation(index())}
                            disabled={skippedHistory().length === 0}
                            title="Previous recommendation"
                          >
                            Back
                          </button>
                          <button
                            class="btn btn-ghost btn-sm btn-nav"
                            onClick={() => skipRecommendation(index())}
                            disabled={fetchingMore()}
                            title="Next recommendation"
                          >
                            {fetchingMore() ? '...' : 'Next'}
                          </button>
                        </Show>
                        <button class="btn btn-primary btn-order" onClick={() => handlePlaceOrder(index())}>
                          Place Order
                        </button>
                      </div>
                      </Show>
                    </div>
                </Show>

                {/* Settings View */}
                <Show when={index() < visibleSlots() && slot.recommendation && !slot.isOrdered && slot.showSettings}>
                    <div class="card-content settings-view">
                      <div class="card-header">
                        <span class="settings-title">Slot Settings</span>
                        <div class="settings-header-actions">
                          <button class="gear-btn" title="Cancel" onClick={() => toggleSettings(index())}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                              <path d="M18 6L6 18M6 6l12 12"/>
                            </svg>
                          </button>
                          <Show when={slotSettingsChanged(slot)}>
                            <button class="gear-btn confirm" title="Apply settings" onClick={() => confirmSlotSettings(index())}>
                              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M20 6L9 17l-5-5"/>
                              </svg>
                            </button>
                          </Show>
                        </div>
                      </div>

                      <div class="settings-controls">
                        <div class="setting-item">
                          <div class="setting-label-row">
                            <label>Style</label>
                            <span class="setting-help" data-tooltip="How often you check the GE. Passive: once a day. Hybrid: a few times. Active: frequently.">?</span>
                          </div>
                          <div class="segmented-control">
                            <button
                              class={`segment ${slot.settings.style === 'passive' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'style', 'passive')}
                            >Passive</button>
                            <button
                              class={`segment ${slot.settings.style === 'hybrid' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'style', 'hybrid')}
                            >Hybrid</button>
                            <button
                              class={`segment ${slot.settings.style === 'active' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'style', 'active')}
                            >Active</button>
                          </div>
                        </div>

                        <div class="setting-item">
                          <div class="setting-label-row">
                            <label>Risk</label>
                            <span class="setting-help" data-tooltip="Higher risk means potentially higher profits but less consistent fills and more volatility.">?</span>
                          </div>
                          <div class="segmented-control">
                            <button
                              class={`segment ${slot.settings.risk === 'low' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'risk', 'low')}
                            >Low</button>
                            <button
                              class={`segment ${slot.settings.risk === 'medium' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'risk', 'medium')}
                            >Medium</button>
                            <button
                              class={`segment ${slot.settings.risk === 'high' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'risk', 'high')}
                            >High</button>
                          </div>
                        </div>

                        <div class="setting-item">
                          <div class="setting-label-row">
                            <label>Margin</label>
                            <span class="setting-help" data-tooltip="Safe: tighter margins, faster fills. Aggro: wider margins, higher profit but slower fills.">?</span>
                          </div>
                          <div class="segmented-control">
                            <button
                              class={`segment ${slot.settings.margin === 'conservative' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'margin', 'conservative')}
                            >Safe</button>
                            <button
                              class={`segment ${slot.settings.margin === 'moderate' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'margin', 'moderate')}
                            >Mid</button>
                            <button
                              class={`segment ${slot.settings.margin === 'aggressive' ? 'active' : ''}`}
                              onClick={() => updateSlotSettings(index(), 'margin', 'aggressive')}
                            >Aggro</button>
                          </div>
                        </div>
                      </div>
                    </div>
                </Show>

                {/* Order Tracking View - Buying Phase */}
                <Show when={index() < visibleSlots() && slot.recommendation && slot.isOrdered && slot.phase === 'buying'}>
                    <div class="card-content tracking-view" data-slot-index={index()}>
                      {/* Update Recommendation Banner - Full Context */}
                      <Show when={slot.pendingUpdate}>
                        <div class={`update-banner-full update-${slot.pendingUpdate!.type.toLowerCase().replace('_', '-')}`}>
                          <div class="update-header">
                            <div class="update-type">
                              <span class="update-icon">
                                {slot.pendingUpdate!.type === 'SWITCH_ITEM' && '⟲'}
                                {slot.pendingUpdate!.type === 'SELL_NOW' && '↓'}
                                {slot.pendingUpdate!.type === 'ADJUST_PRICE' && '↕'}
                              </span>
                              <span class="update-label">
                                {slot.pendingUpdate!.type === 'SWITCH_ITEM' && 'Switch Item'}
                                {slot.pendingUpdate!.type === 'SELL_NOW' && 'Sell Now'}
                                {slot.pendingUpdate!.type === 'ADJUST_PRICE' && 'Adjust Price'}
                              </span>
                            </div>
                            <Show when={slot.pendingUpdate!.profitDelta !== undefined}>
                              <span class={`update-profit ${slot.pendingUpdate!.profitDelta! >= 0 ? 'positive' : 'negative'}`}>
                                {slot.pendingUpdate!.profitDelta! >= 0 ? '+' : ''}{formatGold(slot.pendingUpdate!.profitDelta!)}
                              </span>
                            </Show>
                          </div>
                          <p class="update-reason">{slot.pendingUpdate!.reason}</p>
                          <Show when={slot.pendingUpdate!.type === 'SWITCH_ITEM' && slot.pendingUpdate!.newItem}>
                            <div class="update-new-item">
                              <img src={itemImageUrl(slot.pendingUpdate!.newItem!.itemId)} alt={slot.pendingUpdate!.newItem!.item} loading="lazy" onError={handleImageError} />
                              <span class="new-item-name">{slot.pendingUpdate!.newItem!.item}</span>
                              <span class="new-item-profit">+{formatGold(slot.pendingUpdate!.newItem!.expectedProfit)}</span>
                            </div>
                          </Show>
                          <Show when={slot.pendingUpdate!.type === 'ADJUST_PRICE' && slot.pendingUpdate!.newSellPrice}>
                            <div class="update-price-change">
                              <span class="old-price">{formatGold(slot.pendingUpdate!.originalSellPrice || 0, 'gp')}</span>
                              <span class="price-arrow">→</span>
                              <span class="new-price">{formatGold(slot.pendingUpdate!.newSellPrice!, 'gp')}</span>
                            </div>
                          </Show>
                          <div class="update-actions-full">
                            <button class="btn btn-ghost btn-sm" onClick={() => ignoreUpdate(index())}>Ignore</button>
                            <button class="btn btn-primary btn-sm" onClick={() => acceptUpdate(index())}>Accept</button>
                          </div>
                        </div>
                      </Show>

                      <div class="card-header">
                        <div class="item-icon">
                          <Show when={slot.acceptedUpdate?.type === 'SWITCH_ITEM' && slot.acceptedUpdate?.newItem}>
                            <img
                              src={itemImageUrl(slot.acceptedUpdate!.newItem!.itemId)}
                              alt={slot.acceptedUpdate!.newItem!.item}
                              onError={handleImageError}
                            />
                          </Show>
                          <Show when={!slot.acceptedUpdate || slot.acceptedUpdate.type !== 'SWITCH_ITEM'}>
                            <img
                              src={itemImageUrl(slot.recommendation!.itemId)}
                              alt={slot.recommendation!.item}
                              onError={handleImageError}
                            />
                          </Show>
                        </div>
                        <Show when={slot.acceptedUpdate?.type === 'SWITCH_ITEM' && slot.acceptedUpdate?.newItem}>
                          <div class="item-name tracking-item update-value">{slot.acceptedUpdate!.newItem!.item}</div>
                        </Show>
                        <Show when={!slot.acceptedUpdate || slot.acceptedUpdate.type !== 'SWITCH_ITEM'}>
                          <div class="item-name tracking-item">{slot.recommendation!.item}</div>
                        </Show>
                        <span class="status-badge">Buying</span>
                      </div>

                      <div class={`price-row compact ${slot.acceptedUpdate ? 'update-highlight' : ''}`}>
                        <Show when={slot.acceptedUpdate?.type === 'SWITCH_ITEM' && slot.acceptedUpdate?.newItem}>
                          <span class="price-compact update-value">{formatGold(slot.acceptedUpdate!.newItem!.buyPrice, 'gp')}</span>
                          <span class="price-arrow">→</span>
                          <span class="price-compact update-value">{formatGold(slot.acceptedUpdate!.newItem!.sellPrice, 'gp')}</span>
                        </Show>
                        <Show when={slot.acceptedUpdate?.type === 'SELL_NOW' && slot.acceptedUpdate?.adjustedSellPrice}>
                          <span class="price-compact">{formatGold(slot.recommendation!.buyPrice, 'gp')}</span>
                          <span class="price-arrow">→</span>
                          <span class="price-compact update-value">{formatGold(slot.acceptedUpdate!.adjustedSellPrice!, 'gp')}</span>
                        </Show>
                        <Show when={!slot.acceptedUpdate || (slot.acceptedUpdate.type !== 'SWITCH_ITEM' && slot.acceptedUpdate.type !== 'SELL_NOW')}>
                          <span class="price-compact">{formatGold(slot.recommendation!.buyPrice, 'gp')}</span>
                          <span class="price-arrow">→</span>
                          <span class="price-compact">{formatGold(slot.recommendation!.sellPrice, 'gp')}</span>
                        </Show>
                      </div>

                      <div class="stats-row compact">
                        <div class="stat-item">
                          <span class="stat-value">{slot.quantity.toLocaleString()}</span>
                          <span class="stat-label">Rec Qty</span>
                        </div>
                        <div class="stat-item filled-stat">
                          <div class="filled-value-row">
                            <button class="adj-btn" onClick={() => incrementFilled(index(), -1)}>−</button>
                            <input
                              type="number"
                              class="filled-value-input"
                              value={slot.filled}
                              min={0}
                              max={slot.quantity}
                              onBlur={(e) => updateFilled(index(), parseInt(e.currentTarget.value) || 0)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  updateFilled(index(), parseInt(e.currentTarget.value) || 0);
                                  e.currentTarget.blur();
                                }
                              }}
                            />
                            <button class="adj-btn" onClick={() => incrementFilled(index(), 1)}>+</button>
                          </div>
                          <span class="stat-label">Filled</span>
                        </div>
                      </div>

                      <div class="progress-container">
                        <div
                          class={`progress interactive ${draggingSlot() === index() ? 'dragging' : ''}`}
                          data-progress-slot={index()}
                          onMouseDown={(e) => handleProgressMouseDown(index(), e)}
                        >
                          <div
                            class="progress-bar"
                            style={{ width: `${fillPercent(slot)}%` }}
                          />
                          <div
                            class="progress-handle"
                            style={{ left: `${fillPercent(slot)}%` }}
                          />
                        </div>
                        <span class="progress-text">{Math.round(fillPercent(slot))}%</span>
                      </div>

                      <Show when={!slot.pendingUpdate}>
                        <div class="flex-spacer"></div>

                        {/* Hide sparkline when pending update to make room for notification */}
                        <Show when={showInfoSlot() !== index()}>
                          <div class="sparkline-container">
                            {generateSparkline(slot.recommendation!.priceHistory, slot.recommendation!.trend, slot.settings.style, index())}
                          </div>
                        </Show>

                        <Show when={showInfoSlot() === index()}>
                        <div class="info-panel">
                          <div class="info-row">
                            <Tooltip text="Estimated time for orders to fill based on market activity" position="left">
                              <span class="info-label info-label-tooltip">Time to Fill <InfoIcon /></span>
                            </Tooltip>
                            <span class="info-value">{slot.recommendation!.expectedHours}h</span>
                          </div>
                          <div class="info-row">
                            <Tooltip text="Model confidence in this recommendation. Higher = more certain." position="left">
                              <span class="info-label info-label-tooltip">Confidence <InfoIcon /></span>
                            </Tooltip>
                            <span class="info-value">{formatPercent(slot.recommendation!.confidence)}</span>
                          </div>
                          <div class="info-row">
                            <span class="info-label">Volume 24h</span>
                            <span class="info-value">{formatRsNumber(slot.recommendation!.volume24h, 3)}</span>
                          </div>
                          <div class="info-row">
                            <span class="info-label">Live Price</span>
                            <span class="info-value">{formatRsPrice(slot.recommendation!.buyPrice)}</span>
                          </div>
                        </div>
                        </Show>
                      </Show>

                      <div class="action-row tracking-actions">
                        <button
                          class={`btn btn-ghost btn-sm btn-info ${showInfoSlot() === index() ? 'active' : ''}`}
                          title="More details"
                          onClick={() => setShowInfoSlot(showInfoSlot() === index() ? null : index())}
                        >
                          <svg class="info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="16" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12.01" y2="8"/>
                          </svg>
                        </button>
                        <Show when={confirmingCancel() === index()}>
                          <div class="cancel-confirm">
                            <span class="cancel-confirm-text">Cancel?</span>
                            <button class="btn-confirm-no" onClick={() => setConfirmingCancel(null)}>No</button>
                            <button class="btn-confirm-yes" onClick={() => {
                              cancelOrder(index());
                              setConfirmingCancel(null);
                            }}>Yes</button>
                          </div>
                        </Show>
                        <Show when={confirmingCancel() !== index()}>
                          <button class="btn btn-ghost btn-sm btn-cancel" onClick={() => setConfirmingCancel(index())}>
                            Cancel
                          </button>
                        </Show>
                        <Show when={slot.acceptedUpdate && slot.acceptedUpdate.type !== 'SELL_NOW'}>
                          <button class="btn btn-warning btn-sm" onClick={() => confirmUpdate(index())}>
                            {slot.acceptedUpdate!.type === 'SWITCH_ITEM' ? 'Switch' : 'Update'}
                          </button>
                        </Show>
                        <Show when={slot.acceptedUpdate && slot.acceptedUpdate.type === 'SELL_NOW'}>
                          <button class="btn btn-warning btn-sm" onClick={() => confirmUpdate(index())}>
                            Cancel & Sell
                          </button>
                        </Show>
                        <Show when={!slot.acceptedUpdate}>
                          <button class="btn btn-success btn-sm" onClick={() => startSelling(index())} disabled={slot.filled === 0}>
                            Sell
                          </button>
                        </Show>
                      </div>
                    </div>
                </Show>

                {/* Order Tracking View - Selling Phase */}
                <Show when={index() < visibleSlots() && slot.recommendation && slot.isOrdered && slot.phase === 'selling'}>
                    <div class="card-content tracking-view" data-slot-index={index()}>
                      {/* Update Recommendation Banner - Full Context */}
                      <Show when={slot.pendingUpdate}>
                        <div class={`update-banner-full update-${slot.pendingUpdate!.type.toLowerCase().replace('_', '-')}`}>
                          <div class="update-header">
                            <div class="update-type">
                              <span class="update-icon">
                                {slot.pendingUpdate!.type === 'ADJUST_PRICE' && '↕'}
                              </span>
                              <span class="update-label">Adjust Price</span>
                            </div>
                            <Show when={slot.pendingUpdate!.profitDelta !== undefined}>
                              <span class={`update-profit ${slot.pendingUpdate!.profitDelta! >= 0 ? 'positive' : 'negative'}`}>
                                {slot.pendingUpdate!.profitDelta! >= 0 ? '+' : ''}{formatGold(slot.pendingUpdate!.profitDelta!)}
                              </span>
                            </Show>
                          </div>
                          <p class="update-reason">{slot.pendingUpdate!.reason}</p>
                          <Show when={slot.pendingUpdate!.newSellPrice}>
                            <div class="update-price-change">
                              <span class="old-price">{formatGold(slot.pendingUpdate!.originalSellPrice || slot.recommendation!.sellPrice, 'gp')}</span>
                              <span class="price-arrow">→</span>
                              <span class="new-price">{formatGold(slot.pendingUpdate!.newSellPrice!, 'gp')}</span>
                            </div>
                          </Show>
                          <div class="update-actions-full">
                            <button class="btn btn-ghost btn-sm" onClick={() => ignoreUpdate(index())}>Ignore</button>
                            <button class="btn btn-primary btn-sm" onClick={() => acceptUpdate(index())}>Accept</button>
                          </div>
                        </div>
                      </Show>

                      <div class="card-header">
                        <div class="item-icon">
                          <img
                            src={itemImageUrl(slot.recommendation!.itemId)}
                            alt={slot.recommendation!.item}
                            onError={handleImageError}
                          />
                        </div>
                        <div class="item-name tracking-item">{slot.recommendation!.item}</div>
                        <span class="status-badge selling">Selling</span>
                      </div>

                      <div class={`price-row compact ${slot.acceptedUpdate ? 'update-highlight' : ''}`}>
                        <span class="price-compact">{formatGold(slot.recommendation!.buyPrice, 'gp')}</span>
                        <span class="price-arrow">→</span>
                        <Show when={slot.acceptedUpdate?.type === 'ADJUST_PRICE' && slot.acceptedUpdate?.newSellPrice}>
                          <span class="price-compact sell-price update-value">{formatGold(slot.acceptedUpdate!.newSellPrice!, 'gp')}</span>
                        </Show>
                        <Show when={!slot.acceptedUpdate || slot.acceptedUpdate.type !== 'ADJUST_PRICE'}>
                          <span class="price-compact sell-price">{formatGold(slot.recommendation!.sellPrice, 'gp')}</span>
                        </Show>
                      </div>

                      <div class="stats-row compact">
                        <div class="stat-item">
                          <span class="stat-value">{slot.filled.toLocaleString()}</span>
                          <span class="stat-label">Bought</span>
                        </div>
                        <div class="stat-item filled-stat">
                          <div class="filled-value-row">
                            <button class="adj-btn" onClick={() => incrementSold(index(), -1)}>−</button>
                            <input
                              type="number"
                              class="filled-value-input"
                              value={slot.sold}
                              min={0}
                              max={slot.filled}
                              onBlur={(e) => updateSold(index(), parseInt(e.currentTarget.value) || 0)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  updateSold(index(), parseInt(e.currentTarget.value) || 0);
                                  e.currentTarget.blur();
                                }
                              }}
                            />
                            <button class="adj-btn" onClick={() => incrementSold(index(), 1)}>+</button>
                          </div>
                          <span class="stat-label">Sold</span>
                        </div>
                      </div>

                      <div class="progress-container">
                        <div
                          class={`progress interactive ${draggingSlot() === index() ? 'dragging' : ''}`}
                          data-sell-progress-slot={index()}
                          onMouseDown={(e) => handleSellProgressMouseDown(index(), e)}
                        >
                          <div
                            class="progress-bar sell-progress"
                            style={{ width: `${sellPercent(slot)}%` }}
                          />
                          <div
                            class="progress-handle sell-handle"
                            style={{ left: `${sellPercent(slot)}%` }}
                          />
                        </div>
                        <span class="progress-text">{Math.round(sellPercent(slot))}%</span>
                      </div>

                      {/* Hide flex-spacer and sparkline when pending update to make room for notification */}
                      <Show when={!slot.pendingUpdate}>
                        <div class="flex-spacer"></div>

                        <Show when={showInfoSlot() !== index()}>
                          <div class="sparkline-container">
                            {generateSparkline(slot.recommendation!.priceHistory, slot.recommendation!.trend, slot.settings.style, index())}
                          </div>
                        </Show>

                        <Show when={showInfoSlot() === index()}>
                          <div class="info-panel">
                            <div class="info-row">
                              <span class="info-label">Est. Profit</span>
                              <span class="info-value profit-text">+{formatGold((slot.recommendation!.sellPrice - slot.recommendation!.buyPrice) * slot.sold)}</span>
                            </div>
                            <div class="info-row">
                              <Tooltip text="Model confidence in this recommendation. Higher = more certain." position="left">
                                <span class="info-label info-label-tooltip">Confidence <InfoIcon /></span>
                              </Tooltip>
                              <span class="info-value">{formatPercent(slot.recommendation!.confidence)}</span>
                            </div>
                            <div class="info-row">
                              <Tooltip text="Estimated time for sell orders to fill" position="left">
                                <span class="info-label info-label-tooltip">Time to Sell <InfoIcon /></span>
                              </Tooltip>
                              <span class="info-value">{slot.recommendation!.expectedHours}h</span>
                            </div>
                            <div class="info-row">
                              <span class="info-label">Sell Price</span>
                              <span class="info-value">{formatRsPrice(slot.recommendation!.sellPrice)}</span>
                            </div>
                          </div>
                        </Show>
                      </Show>

                      <div class="action-row tracking-actions">
                        <button
                          class={`btn btn-ghost btn-sm btn-info ${showInfoSlot() === index() ? 'active' : ''}`}
                          title="More details"
                          onClick={() => setShowInfoSlot(showInfoSlot() === index() ? null : index())}
                        >
                          <svg class="info-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="16" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12.01" y2="8"/>
                          </svg>
                        </button>
                        <button class="btn btn-ghost btn-sm" onClick={() => goBackToBuying(index())}>
                          Back
                        </button>
                        <Show when={slot.acceptedUpdate}>
                          <button class="btn btn-warning btn-sm" onClick={() => confirmUpdate(index())}>
                            Update
                          </button>
                        </Show>
                        <Show when={!slot.acceptedUpdate}>
                          <button class="btn btn-success btn-sm" onClick={() => completeOrder(index())}>
                            Complete
                          </button>
                        </Show>
                      </div>
                    </div>
                </Show>

                {/* Empty Slot (unlocked but no recommendation) */}
                <Show when={index() < visibleSlots() && !slot.recommendation && !slot.isOrdered}>
                  <div class="card-content empty-slot">
                    <span class="empty-text">No recommendations</span>
                    <span class="empty-subtext">Check back later</span>
                  </div>
                </Show>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Sign In Modal */}
      <Show when={showSignInModal()}>
        <div class="modal-overlay" onClick={() => setShowSignInModal(false)}>
          <div class="sign-in-modal" onClick={(e) => e.stopPropagation()}>
            <div class="modal-header">
              <h3>Sign in to track trades</h3>
              <button class="modal-close" aria-label="Close modal" onClick={() => setShowSignInModal(false)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>
            </div>
            <div class="modal-body">
              <p class="modal-description">
                Create an account to track your trades, view profit history, and get personalized recommendations.
              </p>
              <div class="sign-in-buttons">
                <button class="btn-oauth btn-google" onClick={handleSignIn}>
                  <svg viewBox="0 0 24 24" width="18" height="18">
                    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                  </svg>
                  Continue with Google
                </button>
                <button class="btn-oauth btn-apple" onClick={handleSignIn}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                    <path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09l.01-.01zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z"/>
                  </svg>
                  Continue with Apple
                </button>
                <button class="btn-oauth btn-discord" onClick={handleSignIn}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                    <path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057 19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028 14.09 14.09 0 0 0 1.226-1.994.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/>
                  </svg>
                  Continue with Discord
                </button>
                <div class="sign-in-divider">
                  <span>or</span>
                </div>
                <button class="btn-oauth btn-email" onClick={handleSignIn}>
                  <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="4" width="20" height="16" rx="2"/>
                    <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>
                  </svg>
                  Sign up with Email
                </button>
              </div>
              <p class="modal-footer-text">
                Free accounts can track up to {TIER_LIMITS.free} trades at once
              </p>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
