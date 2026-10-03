/**
 * GEE BULK RUNNER - v9 (Production-Ready for 612+ Tasks)
 * Fixes: Stops after 300 iterations, skipped tasks
 * Features: Better error handling, progress tracking, resume capability
 */
async function runAllTasks() {
    // --- CONFIGURATION ---
    const DELAY_BETWEEN_TASKS = 3500;  // 3.5 seconds between tasks
    const POPUP_WAIT = 2500;           // 2.5 seconds for popup to load
    const POPUP_RETRY_ATTEMPTS = 5;    // Increased to 5 attempts
    const RETRY_DELAY = 800;           // 800ms between retries
    const SAFETY_PAUSE_EVERY = 50;     // Pause every 50 tasks
    const SAFETY_PAUSE_DURATION = 10000; // 10 second pause
    // ---------------------
    
    console.log("╔════════════════════════════════════════════════════╗");
    console.log("║  GEE BULK TASK RUNNER v9 - Production Ready      ║");
    console.log("║  Optimized for 612+ tasks without interruption   ║");
    console.log("╚════════════════════════════════════════════════════╝\n");
    
    let processedCount = 0;
    let successCount = 0;
    let failedCount = 0;
    let skippedCount = 0;
    
    // 1. ENHANCED SHADOW DOM SEARCH
    function queryShadowDom(root, conditionFn, resultList = []) {
        if (!root) return resultList;
        
        try {
            // Check current element
            if (conditionFn(root)) {
                resultList.push(root);
            }
            
            // Check Shadow Root
            if (root.shadowRoot) {
                queryShadowDom(root.shadowRoot, conditionFn, resultList);
            }
            
            // Check Children
            if (root.children) {
                for (let child of root.children) {
                    queryShadowDom(child, conditionFn, resultList);
                }
            }
        } catch (err) {
            // Silently handle DOM access errors
        }
        
        return resultList;
    }
    
    // 2. FIND ALL TASK LIST BUTTONS (REFRESH EACH TIME)
    function findTaskListButtons() {
        // Always get fresh list to avoid stale references
        return queryShadowDom(document.body, (el) => {
            return el.tagName && 
                   el.tagName.toLowerCase() === 'ee-button' && 
                   el.classList.contains('run-button') &&
                   !el.disabled &&
                   el.offsetParent !== null; // Must be visible
        });
    }
    
    // 3. FIND POPUP RUN BUTTON (Enhanced with more strategies)
    function findPopupRunButton() {
        // Strategy 1: Look for ee-button with "RUN" text that's NOT in task list
        let buttons = queryShadowDom(document.body, (el) => {
            if (!el.tagName || el.tagName.toLowerCase() !== 'ee-button') return false;
            
            // Must be visible
            if (el.offsetParent === null && el.style.display === 'none') return false;
            
            // Get text content
            let text = '';
            if (el.shadowRoot) {
                text = el.shadowRoot.textContent || '';
            }
            text += el.textContent || '';
            text = text.trim().toUpperCase();
            
            // Must contain "RUN" but NOT be a task list button
            return text.includes('RUN') && !el.classList.contains('run-button');
        });
        
        if (buttons.length > 0) {
            return buttons[buttons.length - 1];
        }
        
        // Strategy 2: Look inside dialogs
        const dialogs = document.querySelectorAll('dialog, [role="dialog"], .modal, .popup, ee-dialog');
        for (let dialog of dialogs) {
            if (dialog.offsetParent === null) continue; // Skip hidden dialogs
            
            const dialogButtons = queryShadowDom(dialog, (el) => {
                if (!el.tagName || el.tagName.toLowerCase() !== 'ee-button') return false;
                let text = (el.shadowRoot?.textContent || el.textContent || '').trim().toUpperCase();
                return text.includes('RUN');
            });
            if (dialogButtons.length > 0) {
                return dialogButtons[dialogButtons.length - 1];
            }
        }
        
        // Strategy 3: Look for any visible button with exact "RUN" text
        buttons = queryShadowDom(document.body, (el) => {
            if (!el.tagName) return false;
            const tag = el.tagName.toLowerCase();
            if (tag !== 'button' && tag !== 'ee-button') return false;
            
            // Check visibility with bounding rect
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return false;
            if (rect.top < 0 || rect.left < 0) return false; // Off-screen
            
            let text = (el.shadowRoot?.textContent || el.textContent || '').trim().toUpperCase();
            return text === 'RUN';
        });
        
        if (buttons.length > 0) {
            return buttons[buttons.length - 1];
        }
        
        return null;
    }
    
    // 4. CLICK POPUP RUN BUTTON WITH ENHANCED RETRY
    async function clickPopupRunButton() {
        for (let attempt = 1; attempt <= POPUP_RETRY_ATTEMPTS; attempt++) {
            const popupBtn = findPopupRunButton();
            
            if (popupBtn) {
                try {
                    // Ensure button is in viewport
                    popupBtn.scrollIntoView({ block: 'center', behavior: 'instant' });
                    await new Promise(r => setTimeout(r, 300));
                    
                    // Multiple click attempts
                    popupBtn.click();
                    await new Promise(r => setTimeout(r, 100));
                    popupBtn.dispatchEvent(new MouseEvent('click', { 
                        bubbles: true, 
                        cancelable: true,
                        view: window
                    }));
                    await new Promise(r => setTimeout(r, 100));
                    
                    // Try focusing and clicking again
                    popupBtn.focus();
                    popupBtn.click();
                    
                    return true;
                } catch (err) {
                    console.warn(`   ⚠️  Click attempt ${attempt} error:`, err.message);
                }
            }
            
            // Wait before retry
            if (attempt < POPUP_RETRY_ATTEMPTS) {
                await new Promise(r => setTimeout(r, RETRY_DELAY));
            }
        }
        
        // Final fallback: Enter key
        try {
            const enterEvent = new KeyboardEvent('keydown', {
                bubbles: true,
                cancelable: true,
                keyCode: 13,
                key: 'Enter',
                code: 'Enter'
            });
            document.activeElement?.dispatchEvent(enterEvent);
            document.body.dispatchEvent(enterEvent);
            await new Promise(r => setTimeout(r, 200));
        } catch (err) {
            // Ignore keyboard fallback errors
        }
        
        return false;
    }
    
    // 5. CLOSE ANY POPUP (for cleanup)
    function closePopup() {
        try {
            // Try to find and click Cancel/Close button
            const closeButtons = queryShadowDom(document.body, (el) => {
                if (!el.tagName) return false;
                const tag = el.tagName.toLowerCase();
                if (tag !== 'button' && tag !== 'ee-button') return false;
                
                const text = (el.shadowRoot?.textContent || el.textContent || '').trim().toUpperCase();
                return text.includes('CANCEL') || text.includes('CLOSE');
            });
            
            if (closeButtons.length > 0) {
                closeButtons[closeButtons.length - 1].click();
            }
            
            // Press Escape key
            const escEvent = new KeyboardEvent('keydown', {
                bubbles: true,
                cancelable: true,
                keyCode: 27,
                key: 'Escape',
                code: 'Escape'
            });
            document.body.dispatchEvent(escEvent);
        } catch (err) {
            // Ignore cleanup errors
        }
    }
    
    // 6. MAIN PROCESSING LOOP WITH CONTINUOUS REFRESH
    console.log("🔍 Starting continuous task processing...\n");
    
    while (true) {
        // Get fresh list of tasks each iteration
        const taskListButtons = findTaskListButtons();
        
        if (taskListButtons.length === 0) {
            console.log("\n✅ No more tasks found with RUN buttons!");
            console.log("╔════════════════════════════════════════════════════╗");
            console.log("║                  FINAL SUMMARY                     ║");
            console.log("╠════════════════════════════════════════════════════╣");
            console.log(`║  Total Processed: ${processedCount.toString().padEnd(31)}║`);
            console.log(`║  Successful: ${successCount.toString().padEnd(35)}║`);
            console.log(`║  Failed: ${failedCount.toString().padEnd(39)}║`);
            console.log(`║  Skipped: ${skippedCount.toString().padEnd(38)}║`);
            console.log("╚════════════════════════════════════════════════════╝");
            break;
        }
        
        // Process first available task
        const taskBtn = taskListButtons[0];
        processedCount++;
        
        console.log(`\n[${processedCount}] Processing task...`);
        console.log(`   📊 Remaining: ${taskListButtons.length} tasks with RUN button`);
        
        try {
            // A. Scroll and click task
            taskBtn.scrollIntoView({ block: 'center', behavior: 'instant' });
            await new Promise(r => setTimeout(r, 400));
            
            taskBtn.click();
            console.log("   → Task clicked, waiting for popup...");
            
            // B. Wait for popup
            await new Promise(r => setTimeout(r, POPUP_WAIT));
            
            // C. Click popup Run button
            const success = await clickPopupRunButton();
            
            if (success) {
                console.log(`   ✅ Task ${processedCount} submitted!`);
                successCount++;
            } else {
                console.warn(`   ⚠️  Task ${processedCount} - popup button not found`);
                failedCount++;
                // Try to close popup
                closePopup();
                await new Promise(r => setTimeout(r, 1000));
            }
            
        } catch (err) {
            console.error(`   ❌ Error on task ${processedCount}:`, err.message);
            failedCount++;
            // Try to recover
            closePopup();
            await new Promise(r => setTimeout(r, 1000));
        }
        
        // D. Safety pause every N tasks
        if (processedCount % SAFETY_PAUSE_EVERY === 0) {
            console.log(`\n⏸️  Safety pause at ${processedCount} tasks (${SAFETY_PAUSE_DURATION/1000}s)...`);
            console.log(`   Progress: ${successCount}✅ ${failedCount}❌`);
            await new Promise(r => setTimeout(r, SAFETY_PAUSE_DURATION));
        }
        
        // E. Regular delay
        await new Promise(r => setTimeout(r, DELAY_BETWEEN_TASKS));
    }
    
    console.log("\n🎉 All tasks processed!");
    console.log("\n💡 Check the 'Tasks' tab to verify running exports.");
    console.log("💡 Check console for any tasks that may need manual submission.");
}

// Add error handling wrapper
console.log("Starting GEE Bulk Task Runner...");
console.log("⚠️  IMPORTANT: Keep this tab ACTIVE and VISIBLE!");
console.log("⚠️  Do NOT switch tabs, minimize, or let screen sleep!\n");

runAllTasks().catch(err => {
    console.error("\n❌ CRITICAL ERROR:", err);
    console.error("Stack trace:", err.stack);
    console.log("\n💡 You can re-run the script to continue from where it stopped.");
});