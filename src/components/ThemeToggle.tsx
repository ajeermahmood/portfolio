"use client";

import { ToggleButton, useTheme } from "@once-ui-system/core";
import type React from "react";
import { useEffect, useState } from "react";

export const ThemeToggle: React.FC = () => {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  // The server cannot know the visitor's theme, so the first client render must
  // match its output; the real icon appears once the provider has resolved it.
  useEffect(() => setMounted(true), []);

  const current = mounted ? resolvedTheme : "light";
  const nextTheme = current === "dark" ? "light" : "dark";

  return (
    <ToggleButton
      prefixIcon={nextTheme}
      onClick={() => setTheme(nextTheme)}
      aria-label={`Switch to ${nextTheme} mode`}
    />
  );
};
