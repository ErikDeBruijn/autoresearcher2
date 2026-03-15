import { test, expect } from "@playwright/test";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://dllm-experiment.local:8000";

test.describe("Kanban Dashboard", () => {
  test("shows 5 kanban columns", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Backlog")).toBeVisible();
    await expect(page.getByText("Todo")).toBeVisible();
    await expect(page.getByText("Running")).toBeVisible();
    await expect(page.getByText("Done")).toBeVisible();
    await expect(page.getByText("Reviewed")).toBeVisible();
  });

  test("stats bar shows experiment data", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Experiments:")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText(/% ok/)).toBeVisible();
    await expect(page.getByText("World Model:")).toBeVisible();
  });

  test("can open and close proposal form", async ({ page }) => {
    await page.goto("/");
    await page.getByText("+ New Proposal").click();
    // Form has "Epistemic Intent" label
    await expect(page.getByText("Epistemic Intent")).toBeVisible();
    await expect(
      page.getByPlaceholder("What belief or tension does this address?")
    ).toBeVisible();
    // Close via Cancel
    await page.getByText("Cancel").click();
    await expect(page.getByText("Epistemic Intent")).not.toBeVisible();
  });

  test("can submit a proposal via form", async ({ page }) => {
    await page.goto("/");
    await page.getByText("+ New Proposal").click();

    const intent = `E2E test probe ${Date.now()}`;
    await page
      .getByPlaceholder("What belief or tension does this address?")
      .fill(intent);
    await page.getByPlaceholder("Why is this valuable now?").fill("E2E test");
    await page
      .getByPlaceholder("What would we learn regardless of outcome?")
      .fill("Verify form works");

    await page.getByText("Submit Proposal").click();

    // Form should close and card appears in backlog
    await expect(page.getByText("Epistemic Intent")).not.toBeVisible({
      timeout: 5000,
    });
    // Wait for kanban refresh
    await page.waitForTimeout(6000);
    await expect(page.getByText(intent)).toBeVisible({ timeout: 10000 });
  });

  test("can open world model panel", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "World Model" }).click();
    // Panel shows "Beliefs" heading
    await expect(page.getByText("Beliefs").first()).toBeVisible();
  });

  test("can toggle chat sidebar", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Chat" }).click();
    await expect(page.getByText("Research Chat")).toBeVisible();
    await page.getByRole("button", { name: "Chat" }).click();
    await expect(page.getByText("Research Chat")).not.toBeVisible();
  });

  test("kanban shows live data from API", async ({ page }) => {
    const statsResp = await page.request.get(`${API_URL}/api/stats`);
    const stats = await statsResp.json();

    await page.goto("/");
    await page.waitForTimeout(3000);

    // If there are observations, the stats bar should show the count
    if (stats.total_observations > 0) {
      await expect(
        page.getByText(String(stats.total_observations), { exact: true }).first()
      ).toBeVisible();
    }
  });

  test("API lifecycle: created proposal appears in backlog", async ({
    page,
  }) => {
    const intent = `Lifecycle test ${Date.now()}`;
    const resp = await page.request.post(`${API_URL}/api/proposals`, {
      data: {
        intent,
        rationale: "Test lifecycle",
        expected_learning: "Cards flow",
        intervention_type: "probe",
        intervention_spec: { DEPTH: "8", run_steps: "100" },
      },
    });
    expect(resp.ok()).toBeTruthy();

    await page.goto("/");
    await expect(page.getByText(intent)).toBeVisible({ timeout: 10000 });
  });
});
