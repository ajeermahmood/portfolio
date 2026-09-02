"use client";

import { protectedRoutes } from "@/resources";
import { Button, Column, Flex, Heading, PasswordInput, Spinner } from "@once-ui-system/core";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

interface RouteGuardProps {
  children: React.ReactNode;
}

/**
 * Password protection for the routes listed in protectedRoutes. Only the
 * password check needs a request; deciding it in an effect for every page meant
 * the server rendered a spinner site-wide, which left crawlers with no content.
 *
 * Whether a route is enabled at all is checked server-side instead, by
 * requireRouteEnabled() in the page itself. Doing it here returned a 200 with a
 * "not found" body, because discarding `children` also discards the notFound()
 * they throw.
 */
const RouteGuard: React.FC<RouteGuardProps> = ({ children }) => {
  const pathname = usePathname();

  const passwordRequired = Boolean(
    pathname && protectedRoutes[pathname as keyof typeof protectedRoutes],
  );

  const [password, setPassword] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [checkingAuth, setCheckingAuth] = useState(passwordRequired);

  useEffect(() => {
    if (!passwordRequired) {
      setCheckingAuth(false);
      return;
    }

    let cancelled = false;
    setCheckingAuth(true);
    setIsAuthenticated(false);

    fetch("/api/check-auth")
      .then((response) => {
        if (!cancelled && response.ok) setIsAuthenticated(true);
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setCheckingAuth(false);
      });

    return () => {
      cancelled = true;
    };
  }, [pathname, passwordRequired]);

  const handlePasswordSubmit = async () => {
    const response = await fetch("/api/authenticate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });

    if (response.ok) {
      setIsAuthenticated(true);
      setError(undefined);
    } else {
      setError("Incorrect password");
    }
  };

  if (passwordRequired && !isAuthenticated) {
    if (checkingAuth) {
      return (
        <Flex fillWidth paddingY="128" horizontal="center">
          <Spinner />
        </Flex>
      );
    }

    return (
      <Column paddingY="128" maxWidth={24} gap="24" center>
        <Heading align="center" wrap="balance">
          This page is password protected
        </Heading>
        <Column fillWidth gap="8" horizontal="center">
          <PasswordInput
            id="password"
            label="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            errorMessage={error}
          />
          <Button onClick={handlePasswordSubmit}>Submit</Button>
        </Column>
      </Column>
    );
  }

  return <>{children}</>;
};

export { RouteGuard };
