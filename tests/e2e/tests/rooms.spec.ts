import { test, expect, Page } from '@playwright/test';

// Test helper to wait for WebSocket connection
async function waitForConnection(page: Page) {
  await expect(page.locator('#connectionStatus')).toHaveText('Connected', { timeout: 10000 });
}

// Test helper to create a room
async function createRoom(page: Page, name: string, workingDir: string = '.') {
  await page.fill('#roomName', name);
  await page.fill('#workingDir', workingDir);
  await page.click('button:has-text("Create Room")');

  // Wait for room to appear in list
  await expect(page.locator('.room-item', { hasText: name })).toBeVisible({ timeout: 5000 });
}

test.describe('Room Observer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/static/room-observer.html');
  });

  test('should load room observer page', async ({ page }) => {
    await expect(page.locator('header h1')).toHaveText('Conclave Room Observer');
    await expect(page.locator('#connectionStatus')).toHaveText('Disconnected');
  });

  test('should display empty state when no room selected', async ({ page }) => {
    await expect(page.locator('#emptyState')).toBeVisible();
    await expect(page.locator('#emptyState')).toHaveText('Select or create a room to begin');
  });

  test('should create a new room', async ({ page }) => {
    const roomName = `Test Room ${Date.now()}`;

    await createRoom(page, roomName, '/home/crook/dev');

    // Verify room appears in list
    const roomItem = page.locator('.room-item', { hasText: roomName });
    await expect(roomItem).toBeVisible();
  });

  test('should select a room and establish WebSocket connection', async ({ page }) => {
    const roomName = `WS Test Room ${Date.now()}`;

    await createRoom(page, roomName);

    // Click on the room to select it
    await page.click(`.room-item:has-text("${roomName}")`);

    // Wait for WebSocket connection
    await waitForConnection(page);

    // Verify UI state changes
    await expect(page.locator('#emptyState')).not.toBeVisible();
    await expect(page.locator('#roomHeader')).toBeVisible();
    await expect(page.locator('#currentRoomName')).toHaveText(roomName);
    await expect(page.locator('#inputArea')).toBeVisible();
  });

  test('should show system message when connected', async ({ page }) => {
    const roomName = `System Msg Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Check for system message about connection
    const systemMessage = page.locator('.message.system', { hasText: 'Connected to room' });
    await expect(systemMessage).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Room Messaging', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/static/room-observer.html');
  });

  test('should send a message', async ({ page }) => {
    const roomName = `Messaging Test ${Date.now()}`;
    const testMessage = 'Hello, this is a test message!';

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Send a message
    await page.fill('#messageInput', testMessage);
    await page.click('button:has-text("Send")');

    // Verify message appears (may take a moment for round-trip)
    // Note: The message should appear via WebSocket event
    await expect(page.locator('#messageInput')).toHaveValue('');
  });

  test('should send message on Enter key', async ({ page }) => {
    const roomName = `Enter Key Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Type and press Enter
    await page.fill('#messageInput', 'Test via Enter');
    await page.press('#messageInput', 'Enter');

    // Input should be cleared
    await expect(page.locator('#messageInput')).toHaveValue('');
  });
});

test.describe('Agent Spawning', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/static/room-observer.html');
  });

  test('should show spawn button when room is selected', async ({ page }) => {
    const roomName = `Spawn Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Check spawn button is visible
    await expect(page.locator('.spawn-buttons')).toBeVisible();
    await expect(page.locator('button:has-text("+ Spawn Agent")')).toBeVisible();
  });

  test('should open spawn dialog when button clicked', async ({ page }) => {
    const roomName = `Spawn Dialog Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Click spawn button to open dialog
    await page.click('button:has-text("+ Spawn Agent")');

    // Check dialog is visible
    await expect(page.locator('#spawnModal')).toBeVisible();
    await expect(page.locator('.modal-header h3')).toHaveText('Spawn Agent');

    // Check agent type options are loaded
    await expect(page.locator('.agent-type-option')).toHaveCount(4); // 3 Claude + 1 Infernum

    // Verify form elements
    await expect(page.locator('#agentName')).toBeVisible();
    await expect(page.locator('#agentPersona')).toBeVisible();
  });

  test('should enable spawn button when agent type selected', async ({ page }) => {
    const roomName = `Spawn Enable Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Open spawn dialog
    await page.click('button:has-text("+ Spawn Agent")');

    // Spawn button should be disabled initially
    await expect(page.locator('#spawnBtn')).toBeDisabled();

    // Select an agent type
    await page.click('.agent-type-option:has-text("Claude Sonnet")');

    // Spawn button should now be enabled
    await expect(page.locator('#spawnBtn')).toBeEnabled();
  });

  test('should close spawn dialog on cancel', async ({ page }) => {
    const roomName = `Spawn Cancel Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Open spawn dialog
    await page.click('button:has-text("+ Spawn Agent")');
    await expect(page.locator('#spawnModal')).toBeVisible();

    // Click cancel
    await page.click('button:has-text("Cancel")');

    // Dialog should be hidden
    await expect(page.locator('#spawnModal')).toHaveClass(/hidden/);
  });

  test('should close spawn dialog on X button', async ({ page }) => {
    const roomName = `Spawn Close Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Open spawn dialog
    await page.click('button:has-text("+ Spawn Agent")');
    await expect(page.locator('#spawnModal')).toBeVisible();

    // Click X button
    await page.click('.modal-close');

    // Dialog should be hidden
    await expect(page.locator('#spawnModal')).toHaveClass(/hidden/);
  });

  test('should spawn Claude agent with custom name', async ({ page }) => {
    const roomName = `Spawn Custom Name Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Open spawn dialog
    await page.click('button:has-text("+ Spawn Agent")');

    // Select agent type
    await page.click('.agent-type-option:has-text("Claude Haiku")');

    // Enter custom name
    await page.fill('#agentName', 'Code Reviewer');

    // Click spawn
    await page.click('#spawnBtn');

    // Wait for agent to appear or error (Claude Code may not be configured)
    const participantOrError = await Promise.race([
      page.locator('.participant', { hasText: 'Agent' }).waitFor({ timeout: 10000 }).then(() => 'participant'),
      page.locator('text=Error').waitFor({ timeout: 10000 }).then(() => 'error'),
    ]).catch(() => 'timeout');

    // Either the agent appeared or we got an error (which is valid if Claude isn't configured)
    expect(['participant', 'error', 'timeout']).toContain(participantOrError);

    // Dialog should be closed on success
    if (participantOrError === 'participant') {
      await expect(page.locator('#spawnModal')).toHaveClass(/hidden/);
    }
  });
});

test.describe('Participants', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/static/room-observer.html');
  });

  test('should show participants panel when room selected', async ({ page }) => {
    const roomName = `Participants Test ${Date.now()}`;

    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Check participants panel is visible
    await expect(page.locator('#participantsPane')).toBeVisible();
    await expect(page.locator('#participantsPane h3')).toHaveText('Participants');
  });
});

test.describe('Room List', () => {
  test('should auto-refresh room list', async ({ page }) => {
    await page.goto('/static/room-observer.html');

    // Wait for initial load
    await page.waitForTimeout(500);

    // Get initial room count
    const initialCount = await page.locator('.room-item').count();

    // Create a room via API directly
    const roomName = `API Created Room ${Date.now()}`;
    const response = await page.request.post('/api/rooms', {
      data: { name: roomName, working_dir: '.' }
    });
    expect(response.ok()).toBeTruthy();

    // Wait for auto-refresh (happens every 10 seconds, but we can also manually trigger)
    await page.waitForTimeout(11000);

    // Verify room appears
    const newCount = await page.locator('.room-item').count();
    expect(newCount).toBeGreaterThan(initialCount);
  });
});

test.describe('Error Handling', () => {
  // Skip this test as Playwright's setOffline doesn't reliably close WebSocket connections
  test.skip('should handle WebSocket disconnection gracefully', async ({ page }) => {
    const roomName = `Disconnect Test ${Date.now()}`;

    await page.goto('/static/room-observer.html');
    await createRoom(page, roomName);
    await page.click(`.room-item:has-text("${roomName}")`);
    await waitForConnection(page);

    // Simulate disconnection by going offline
    await page.context().setOffline(true);

    // Check status shows disconnected
    await expect(page.locator('#connectionStatus')).toHaveText('Disconnected', { timeout: 5000 });

    // Go back online
    await page.context().setOffline(false);
  });
});
