"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { routes, protectedRoutes } from "@/resources";
import { Flex, Spinner, Button, Heading, Column, PasswordInput } from "@once-ui-system/core";
import NotFound from "@/app/not-found";

interface RouteGuardProps {
  children: React.ReactNode;
}

/**
 * Whether a route is enabled comes from static config, so it is knowable during
 * render. Only the password check needs a request, and only for the routes
 * listed in protectedRoutes. Deciding both in an effect meant the server
 * rendered a spinner for every page, which left crawlers with no content.
 */
function isRouteEnabled(pathname: string | null): boolean {
  if (!pathname) return false;

  if (pathname in routes) {
    return routes[pathname as keyof typeof routes];
  }

  const dynamicRoutes = ["/blog", "/work"] as const;
  for (const route of dynamicRoutes) {
    if (pathname.startsWith(route) && routes[route]) {
      return true;
    }
  }

  return false;
}

const RouteGuard: React.FC<RouteGuardProps> = ({ children }) => {
  const pathname = usePathname();

  const routeEnabled = isRouteEnabled(pathname);
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

  if (!routeEnabled) {
    return <NotFound />;
  }

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
