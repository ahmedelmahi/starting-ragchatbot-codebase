# Frontend Changes: Theme Toggle Button & Light Theme

## Overview
Added a dark/light mode toggle button to the Course Materials Assistant application with a comprehensive light theme variant.

## Files Modified

### 1. `frontend/index.html`
- Added theme toggle button element directly after `<body>` tag
- Button includes:
  - Sun icon SVG (displayed in dark mode)
  - Moon icon SVG (displayed in light mode)
  - ARIA label for screen readers
  - Title attribute for tooltip

### 2. `frontend/style.css`
- **Dark Theme Variables** (`:root`):
  - Added `--code-bg`, `--scrollbar-track`, `--scrollbar-thumb`, `--scrollbar-thumb-hover`
  - Added `--error-bg`, `--error-text`, `--success-bg`, `--success-text`
  - Added `--source-link-bg`, `--source-link-bg-hover`, `--source-link-text-hover`
- **Light Theme Variables** (`[data-theme="light"]`):
  - Light background colors (#f8fafc, #ffffff)
  - Dark text for good contrast (#0f172a, #475569)
  - Adjusted border color (#cbd5e1) for visibility
  - Proper surface hover state (#f1f5f9)
  - Theme-appropriate code block backgrounds
  - Light-friendly error/success colors
  - Theme-appropriate source link colors
- Updated `.message-content code` and `.message-content pre` to use `--code-bg` variable
- Updated `.error-message` and `.success-message` to use theme variables
- Updated `.source-link` and `.source-text` to use theme variables
- Added `.theme-toggle` styles:
  - Fixed position in top-right corner (1rem from edges)
  - Circular button with 44px diameter (40px on mobile)
  - Hover, focus, and active states
  - Icon display logic (sun/moon swap)
- Added `@keyframes iconRotate` animation for smooth icon transitions
- Added transition properties to major UI elements for smooth theme switching
- Added responsive styles for mobile devices

### 3. `frontend/script.js`
- Added `themeToggle` to DOM elements
- Added `initializeTheme()` function:
  - Checks localStorage for saved theme preference
  - Falls back to system color scheme preference
  - Sets up event listeners for click and keyboard
  - Listens for system theme changes
- Added `toggleTheme()` function:
  - Toggles between light and dark themes
  - Saves preference to localStorage
  - Updates ARIA label dynamically
  - Triggers icon rotation animation
- Added `handleThemeKeydown()` function for keyboard accessibility (Enter/Space keys)

## Implementation Details

### Theme Switching Mechanism
- Uses `data-theme` attribute on `<html>` element (`document.documentElement`)
- Dark theme is default (no `data-theme` attribute)
- Light theme activated via `data-theme="light"`
- CSS custom properties (variables) change based on attribute selector

### JavaScript Functionality
- **Toggle on click**: `toggleTheme()` function handles click events
- **Smooth transitions**: CSS transitions (0.3s) applied to all major UI elements
- **State persistence**: Theme saved to `localStorage`
- **System preference**: Respects `prefers-color-scheme` media query
- **Real-time updates**: Listens for system theme changes

### CSS Architecture
- All colors defined as CSS variables in `:root` and `[data-theme="light"]`
- No hardcoded colors outside of variable definitions
- Smooth 0.3s transitions on background-color, border-color, and color properties
- Visual hierarchy maintained through consistent contrast ratios

## Features

### Design
- Fits existing dark theme aesthetic
- Uses primary blue (#2563eb) as accent color in both themes
- Smooth 0.3s transitions for all theme changes
- Icon rotation animation when toggling

### Accessibility
- Fully keyboard navigable (Tab, Enter, Space)
- ARIA label updates dynamically based on current theme
- Focus ring visible for keyboard users
- Title attribute provides tooltip on hover
- **WCAG Compliant Contrast Ratios**:
  - Primary text on background: ~16:1 (exceeds AAA)
  - Secondary text on background: ~7:1 (meets AAA)
  - Primary text on surface: ~18:1 (exceeds AAA)
  - White text on primary buttons: ~4.5:1 (meets AA)

### Persistence
- Theme preference saved to localStorage
- Respects system color scheme preference when no preference is saved
- Responds to real-time system theme changes

### Responsive
- Smaller button size on mobile (40px vs 44px)
- Maintains fixed position on all screen sizes
- High z-index (1000) ensures visibility above content

## Color Palette

### Dark Theme (default)
| Property | Value | Description |
|----------|-------|-------------|
| Background | #0f172a | Deep navy base |
| Surface | #1e293b | Card/sidebar background |
| Surface Hover | #334155 | Interactive surface state |
| Text Primary | #f1f5f9 | Main text color |
| Text Secondary | #94a3b8 | Muted text |
| Border | #334155 | Subtle dividers |
| Code Background | rgba(0,0,0,0.2) | Code block background |
| Error Text | #f87171 | Error messages |
| Success Text | #4ade80 | Success messages |
| Source Link Hover | #818cf8 | Link hover state |

### Light Theme
| Property | Value | Description |
|----------|-------|-------------|
| Background | #f8fafc | Light gray base |
| Surface | #ffffff | White cards/sidebar |
| Surface Hover | #f1f5f9 | Interactive surface state |
| Text Primary | #0f172a | Dark text for contrast |
| Text Secondary | #475569 | Muted but readable |
| Border | #cbd5e1 | Visible dividers |
| Code Background | #f1f5f9 | Light code block background |
| Error Text | #dc2626 | Darker red for light mode |
| Success Text | #16a34a | Darker green for light mode |
| Source Link Hover | #1d4ed8 | Link hover state |

## CSS Variables Reference

```css
/* Dark Theme */
:root {
  --primary-color: #2563eb;
  --primary-hover: #1d4ed8;
  --background: #0f172a;
  --surface: #1e293b;
  --surface-hover: #334155;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --border-color: #334155;
  --user-message: #2563eb;
  --assistant-message: #374151;
  --code-bg: rgba(0, 0, 0, 0.2);
  --error-bg: rgba(239, 68, 68, 0.1);
  --error-text: #f87171;
  --success-bg: rgba(34, 197, 94, 0.1);
  --success-text: #4ade80;
  --source-link-bg: rgba(99, 102, 241, 0.1);
  --source-link-bg-hover: rgba(99, 102, 241, 0.2);
  --source-link-text-hover: #818cf8;
}

/* Light Theme */
[data-theme="light"] {
  --background: #f8fafc;
  --surface: #ffffff;
  --surface-hover: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --border-color: #cbd5e1;
  --assistant-message: #e2e8f0;
  --code-bg: #f1f5f9;
  --error-text: #dc2626;
  --success-text: #16a34a;
  --source-link-bg: rgba(37, 99, 235, 0.1);
  --source-link-bg-hover: rgba(37, 99, 235, 0.15);
  --source-link-text-hover: #1d4ed8;
}
```

## Elements with Theme Support

All existing UI elements work correctly in both themes:
- Chat messages (user and assistant)
- Sidebar and navigation
- Input field and send button
- Course stats and suggested questions
- Source links and collapsible sections
- Code blocks and markdown content
- Error and success messages
- Scrollbars
- Welcome message
