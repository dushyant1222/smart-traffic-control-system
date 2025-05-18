package com.example.Smart_Signal;

public class TimerResponse {
    private int vehicleCount;
    private int signalTime;
    private String activeSide;

    public TimerResponse() {} // NEW

    public TimerResponse(int vehicleCount, int signalTime){
        this.vehicleCount = vehicleCount;
        this.signalTime = signalTime;
    }

    public int getVehicleCount() {
        return vehicleCount;
    }

    public void setVehicleCount(int vehicleCount) {
        this.vehicleCount = vehicleCount;
    }

    public int getSignalTime() {
        return signalTime;
    }

    public void setSignalTime(int signalTime) {
        this.signalTime = signalTime;
    }

    public String getActiveSide() {
        return activeSide;
    }

    public void setActiveSide(String activeSide) {
        this.activeSide = activeSide;
    }
}
