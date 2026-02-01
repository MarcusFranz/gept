import { createSignal, createEffect, Show, type JSX } from 'solid-js';
import Onboarding from './Onboarding';

interface OnboardingWrapperProps {
  children: JSX.Element;
}

export default function OnboardingWrapper(props: OnboardingWrapperProps) {
  const [showOnboarding, setShowOnboarding] = createSignal(false);
  const [loading, setLoading] = createSignal(true);

  // Check if user needs onboarding on mount
  createEffect(async () => {
    try {
      const response = await fetch('/api/settings');
      if (!response.ok) {
        setLoading(false);
        return;
      }

      const data = await response.json();
      if (data.success && data.data?.user) {
        // Show onboarding if tutorial not completed
        if (!data.data.user.tutorialCompleted) {
          setShowOnboarding(true);
        }
      }
    } catch {
      // On error, just show the main content
    } finally {
      setLoading(false);
    }
  });

  const handleComplete = async () => {
    try {
      // Mark tutorial as completed
      await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tutorialCompleted: true
        })
      });
    } catch {
      // Continue even if save fails
    }

    setShowOnboarding(false);
  };

  const handleSkip = async () => {
    try {
      // Mark tutorial as completed without changing capital
      await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tutorialCompleted: true })
      });
    } catch {
      // Continue even if save fails
    }

    setShowOnboarding(false);
  };

  return (
    <>
      <Show when={showOnboarding()}>
        <Onboarding
          onComplete={handleComplete}
          onSkip={handleSkip}
        />
      </Show>
      <Show when={!loading()}>
        {props.children}
      </Show>
    </>
  );
}
