package com.example.Smart_Signal;

import org.springframework.web.bind.annotation.*;

import org.springframework.beans.factory.annotation.Autowired;

@CrossOrigin(origins = "http://localhost:3000")
@RestController
@RequestMapping("/api/traffic")
public class TrafficController {

    @Autowired
    private VehicleService vehicleService;

    @PostMapping("/status")
    public TimerResponse getTrafficStatus(@RequestBody VehicleRequest request) {
        int vehicleCount = request.getVehicleCount();
        int timer = vehicleService.calculateTimer(vehicleCount);
        return new TimerResponse(vehicleCount, timer);
    }

    // NEW: Accepts all 4 vehicle counts
    @PostMapping("/cycle/start")
    public String startCycle(@RequestBody TrafficCycleRequest request) {
        vehicleService.loadVehicleCounts(
                request.getSideA(),
                request.getSideB(),
                request.getSideC(),
                request.getSideD()
        );
        return "Traffic signal cycle started.";
    }

    // NEW: Returns current signal info
    @GetMapping("/cycle/status")
    public TimerResponse getSignalStatus() {
        return vehicleService.getCurrentStatus();
    }
}
