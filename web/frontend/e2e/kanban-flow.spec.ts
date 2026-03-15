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

    // Cleanup: find and delete the test proposal
    const queueResp = await page.request.get(`${API_URL}/api/queue`);
    const queue = await queueResp.json();
    for (const p of queue.backlog || []) {
      if (p.intent === intent) {
        await page.request.delete(`${API_URL}/api/proposals/${p.id}`);
      }
    }
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

  test("chat responds to a question", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Chat" }).click();
    await expect(page.getByText("Research Chat")).toBeVisible();

    // Type and send a message
    const input = page.getByPlaceholder("Ask about the research...");
    await input.fill("How many experiments have been run?");
    await input.press("Enter");

    // Should show "Thinking..." loading state
    await expect(page.getByText("Thinking...")).toBeVisible({ timeout: 5000 });

    // Should get a real response (not Thinking...) within 60 seconds
    // The response should contain something about experiments or beliefs
    await expect(page.getByText("Thinking...")).not.toBeVisible({
      timeout: 60000,
    });

    // Verify there's an assistant response (second message div)
    const responses = page.locator(".bg-gray-800.rounded-lg");
    await expect(responses.first()).toBeVisible();
  });

  test("running column shows worker slots", async ({ page }) => {
    await page.goto("/");
    await page.waitForTimeout(3000);

    // Should show at least one worker slot (name includes 'worker_')
    await expect(page.getByText(/worker_/).first()).toBeVisible({ timeout: 10000 });
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

    // Cleanup: delete the test proposal
    const body = await resp.json();
    await page.request.delete(`${API_URL}/api/proposals/${body.id}`);
  });

  test("drag proposal from backlog to todo", async ({ page }) => {
    // Create a test proposal via API
    const intent = `Drag test ${Date.now()}`;
    const resp = await page.request.post(`${API_URL}/api/proposals`, {
      data: {
        intent,
        rationale: "Test drag",
        expected_learning: "Drag works",
        intervention_type: "probe",
        intervention_spec: { DEPTH: "8" },
      },
    });
    const body = await resp.json();

    await page.goto("/");
    await page.waitForTimeout(6000);
    await expect(page.getByText(intent)).toBeVisible({ timeout: 10000 });

    // Find the draggable card and the todo column drop target
    const card = page.locator(`[data-proposal-id="${body.id}"]`);
    const todoColumn = page.locator('[data-drop-stage="todo"]');

    await card.dragTo(todoColumn);

    // Wait for refresh and verify it moved
    await page.waitForTimeout(6000);
    // The card should now be inside the todo column
    const todoCards = todoColumn.getByText(intent);
    await expect(todoCards).toBeVisible({ timeout: 10000 });

    // Cleanup: delete
    await page.request.delete(`${API_URL}/api/proposals/${body.id}`);
  });

  test("drag proposal from todo back to backlog", async ({ page }) => {
    // Create and promote a test proposal
    const intent = `Drag back test ${Date.now()}`;
    const resp = await page.request.post(`${API_URL}/api/proposals`, {
      data: {
        intent,
        rationale: "Test drag back",
        expected_learning: "Drag back works",
        intervention_type: "probe",
        intervention_spec: { DEPTH: "8" },
      },
    });
    const body = await resp.json();
    // Promote to todo
    await page.request.post(
      `${API_URL}/api/proposals/${body.id}/promote?target_stage=todo`
    );

    await page.goto("/");
    await page.waitForTimeout(6000);

    const card = page.locator(`[data-proposal-id="${body.id}"]`);
    const backlogColumn = page.locator('[data-drop-stage="backlog"]');

    await card.dragTo(backlogColumn);

    await page.waitForTimeout(6000);
    const backlogCards = backlogColumn.getByText(intent);
    await expect(backlogCards).toBeVisible({ timeout: 10000 });

    // Cleanup
    await page.request.delete(`${API_URL}/api/proposals/${body.id}`);
  });

  test("drag proposal to trash deletes it", async ({ page }) => {
    // Create a test proposal
    const intent = `Trash test ${Date.now()}`;
    const resp = await page.request.post(`${API_URL}/api/proposals`, {
      data: {
        intent,
        rationale: "Test trash",
        expected_learning: "Trash works",
        intervention_type: "probe",
        intervention_spec: { DEPTH: "8" },
      },
    });
    const body = await resp.json();

    await page.goto("/");
    await page.waitForTimeout(6000);
    await expect(page.getByText(intent)).toBeVisible({ timeout: 10000 });

    const card = page.locator(`[data-proposal-id="${body.id}"]`);
    const trashZone = page.locator('[data-drop-stage="trash"]');
    await card.dragTo(trashZone);

    // Wait for refresh — card should be gone
    await page.waitForTimeout(6000);
    await expect(page.getByText(intent)).not.toBeVisible({ timeout: 10000 });
  });

  test("delete proposal via API", async ({ page }) => {
    const intent = `Delete API test ${Date.now()}`;
    const resp = await page.request.post(`${API_URL}/api/proposals`, {
      data: {
        intent,
        rationale: "Test delete",
        expected_learning: "Delete works",
        intervention_type: "probe",
        intervention_spec: { DEPTH: "8" },
      },
    });
    const body = await resp.json();

    const delResp = await page.request.delete(
      `${API_URL}/api/proposals/${body.id}`
    );
    expect(delResp.ok()).toBeTruthy();

    // Verify it's gone
    const checkResp = await page.request.delete(
      `${API_URL}/api/proposals/${body.id}`
    );
    expect(checkResp.status()).toBe(404);
  });
});
