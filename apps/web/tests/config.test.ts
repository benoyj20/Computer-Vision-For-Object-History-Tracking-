import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_API_URL, resolveApiUrl } from "../lib/config";

test("uses the local default when the URL is unset or blank", () => {
  assert.equal(resolveApiUrl(undefined), DEFAULT_API_URL);
  assert.equal(resolveApiUrl("   "), DEFAULT_API_URL);
});

test("removes trailing slashes so paths can be appended", () => {
  assert.equal(
    resolveApiUrl("http://localhost:8000//"),
    "http://localhost:8000",
  );
  assert.equal(
    resolveApiUrl(" https://kitchen.example.edu/api/ "),
    "https://kitchen.example.edu/api",
  );
});

test("rejects values that are not http(s) URLs", () => {
  for (const value of ["localhost:8000", "ftp://localhost", "not a url"]) {
    assert.throws(
      () => resolveApiUrl(value),
      /OBJHIST_API_URL must be an http\(s\) URL/,
    );
  }
});
