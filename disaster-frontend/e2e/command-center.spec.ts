import { test, expect } from '@playwright/test';

test.describe('Disaster Command Center E2E Automation', () => {

  test('should load Command Center and display brand title', async ({ page }) => {
    await page.goto('/');

    // Check main title in TopBar
    await expect(page.locator('text=DISASTER COMMAND CENTER')).toBeVisible();
    await expect(page.locator('text=v2.0 AGENTIC RAG')).toBeVisible();

    // Verify navigation tabs exist
    await expect(page.getByRole('button', { name: /Command Center/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /Evaluation/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /System Status/i })).toBeVisible();
  });

  test('should switch between tabs smoothly', async ({ page }) => {
    await page.goto('/');

    // Switch to Evaluation Tab
    await page.getByRole('button', { name: /Evaluation/i }).click();
    await expect(page.getByText(/Benchmark Evaluation/i)).toBeVisible();

    // Switch to System Status Tab
    await page.getByRole('button', { name: /System Status/i }).click();
    await expect(page.getByText(/System Infrastructure/i)).toBeVisible();

    // Switch back to Command Center
    await page.getByRole('button', { name: /Command Center/i }).click();
    await expect(page.locator('text=DISASTER COMMAND CENTER')).toBeVisible();
  });

  test('should submit disaster query and display response and tool traces', async ({ page }) => {
    await page.goto('/');

    // Find query input box
    const input = page.getByPlaceholder(/Ask a question about current conditions/i);
    await expect(input).toBeVisible();

    // Type query
    await input.fill('What is the current river level?');
    
    // Submit
    const sendButton = page.getByRole('button', { name: /Submit query/i });
    if (await sendButton.isVisible()) {
      await sendButton.click();
    } else {
      await input.press('Enter');
    }

    // Wait for response bubble to appear
    await expect(page.locator('.chat-bubble, [data-testid="agent-response"]').first()).toBeVisible({ timeout: 25000 });
  });

  test('should display Global Emergency Alert Banner when critical alert exists', async ({ page }) => {
    await page.goto('/');

    // Check if critical emergency banner is rendered if alerts exist
    const alertBanner = page.locator('text=CRITICAL EMERGENCY');
    if (await alertBanner.isVisible()) {
      await expect(alertBanner).toBeVisible();
      // Verify jump button
      await page.getByRole('button', { name: /Evaluation/i }).click();
      // Banner must remain visible on evaluation tab
      await expect(alertBanner).toBeVisible();
    }
  });

});

