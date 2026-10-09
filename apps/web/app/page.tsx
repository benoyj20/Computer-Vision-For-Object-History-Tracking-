import { historyApiUrl } from "@/lib/server-config";

export default async function HomePage() {
  const apiUrl = await historyApiUrl();

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col gap-6 px-4 py-10 sm:px-6">
      <header className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold">Kitchen History</h1>
        <p className="text-muted-foreground">
          Ask what is on the counter, where an item is, and how long it has been
          out.
        </p>
      </header>

      <section
        aria-labelledby="assistant-status"
        className="rounded-xl border border-border bg-card p-5"
      >
        <h2 id="assistant-status" className="text-lg font-semibold">
          Assistant not available yet
        </h2>
        <p className="mt-2 text-muted-foreground">
          The history service and assistant are still being built. This page
          will connect to{" "}
          <code className="break-all text-foreground">{apiUrl}</code> once they
          exist.
        </p>
      </section>
    </main>
  );
}
