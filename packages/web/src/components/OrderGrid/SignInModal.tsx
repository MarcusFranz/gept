import type { JSX } from 'solid-js';

interface SignInModalProps {
  onClose: () => void;
  onSignIn: () => void;
}

export default function SignInModal(props: SignInModalProps): JSX.Element {
  return (
    <div class="modal-overlay" onClick={props.onClose}>
      <div class="modal" onClick={(e) => e.stopPropagation()}>
        <div class="modal-header">
          <h3>Sign In Required</h3>
          <button class="modal-close" onClick={props.onClose}>×</button>
        </div>
        <div class="modal-body">
          <p>Sign in to place orders and track your trades.</p>
          <div class="modal-actions">
            <button class="btn btn-secondary" onClick={props.onClose}>
              Cancel
            </button>
            <a href="/login" class="btn btn-primary" onClick={props.onSignIn}>
              Sign In
            </a>
          </div>
        </div>
      </div>

      <style>{`
        .modal-overlay {
          position: fixed;
          inset: 0;
          background-color: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: var(--space-4);
        }

        .modal {
          background-color: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          width: 100%;
          max-width: 400px;
          overflow: hidden;
        }

        .modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-4);
          border-bottom: 1px solid var(--border);
        }

        .modal-header h3 {
          margin: 0;
          font-size: var(--font-size-lg);
        }

        .modal-close {
          background: none;
          border: none;
          font-size: 24px;
          color: var(--text-muted);
          cursor: pointer;
          line-height: 1;
        }

        .modal-close:hover {
          color: var(--text-primary);
        }

        .modal-body {
          padding: var(--space-4);
        }

        .modal-body p {
          margin-bottom: var(--space-4);
          color: var(--text-secondary);
        }

        .modal-actions {
          display: flex;
          gap: var(--space-3);
          justify-content: flex-end;
        }
      `}</style>
    </div>
  );
}
