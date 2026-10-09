import { z } from "zod";

/** Local address of the FastAPI history service when OBJHIST_API_URL is unset. */
export const DEFAULT_API_URL = "http://127.0.0.1:8000";

const apiUrlSchema = z.url({ protocol: /^https?$/ });

/**
 * Validates the history service base URL and removes trailing slashes so
 * callers can append paths such as `/history`. Invalid values fail loudly
 * instead of silently falling back to the default.
 */
export function resolveApiUrl(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) {
    return DEFAULT_API_URL;
  }
  const parsed = apiUrlSchema.safeParse(trimmed);
  if (!parsed.success) {
    throw new Error(
      `OBJHIST_API_URL must be an http(s) URL; received "${trimmed}".`,
    );
  }
  return parsed.data.replace(/\/+$/, "");
}
