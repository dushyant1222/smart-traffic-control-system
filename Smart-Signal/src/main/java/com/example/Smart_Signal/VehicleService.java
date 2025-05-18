package com.example.Smart_Signal;

import java.util.concurrent.atomic.AtomicInteger;

import org.springframework.stereotype.Service;

@Service
public class VehicleService {

    private final String[] sides = {"A", "B", "C", "D"};
    private final int[] vehicleCounts = new int[4];
    private final int[] signalTimers = new int[4];
    private AtomicInteger activeIndex = new AtomicInteger(0);

    private int currentTimer = 0;
    private boolean isRunning = false;

    // Used for normal one-time detection
    public int calculateTimer(int vehicleCount) {
        if (!isRunning) {
            int baseTime = (int) Math.ceil(vehicleCount * 2.5);
            currentTimer = Math.max(Math.min(baseTime, 90),2);
            isRunning = true;

            new Thread(() -> {
                try {
                    while (currentTimer > 0) {
                        Thread.sleep(1000); 
                        currentTimer--;
                    }
                    isRunning = false;
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }).start();
        }

        return currentTimer; // ⚠️ Might already be reduced if delay occurred
    }

    // NEW: Load 4 vehicle counts and initialize the cycle
    public void loadVehicleCounts(int a, int b, int c, int d) {
        vehicleCounts[0] = a;
        vehicleCounts[1] = b;
        vehicleCounts[2] = c;
        vehicleCounts[3] = d;

        for (int i = 0; i < 4; i++) {
            signalTimers[i] = Math.min((int) Math.ceil(vehicleCounts[i] * 2.5), 90);
        }

        if (!isRunning) {
            startCycle();
        }
    }

    // NEW: Cycle logic to switch sides every X seconds
    private void startCycle() {
        isRunning = true;

        new Thread(() -> {
            while (true) {
                int idx = activeIndex.get();
                currentTimer = signalTimers[idx];

                while (currentTimer > 0) {
                    try {
                        Thread.sleep(1000);
                        currentTimer--;
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

                activeIndex.set((idx + 1) % 4); 
            }
        }).start();
    }

    // NEW: Returns current active signal info
    public TimerResponse getCurrentStatus() {
        int idx = activeIndex.get();
        TimerResponse response = new TimerResponse();
        response.setVehicleCount(vehicleCounts[idx]);
        response.setSignalTime(currentTimer);
        response.setActiveSide(sides[idx]);
        return response;
    }
}
