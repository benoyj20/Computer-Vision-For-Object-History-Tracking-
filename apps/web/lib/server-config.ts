import "server-only";
import { connection } from "next/server";
import { resolveApiUrl } from "./config";

/**
 * History service URL, read on the server so the browser never sees configuration.
 * `connection()` defers rendering to request time; otherwise the build would bake
 * in whatever OBJHIST_API_URL was set when `next build` ran.
 */
export async function historyApiUrl(): Promise<string> {
  await connection();
  return resolveApiUrl(process.env.OBJHIST_API_URL);
}
