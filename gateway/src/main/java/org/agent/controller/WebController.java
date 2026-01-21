package org.agent.controller;

import org.agent.service.WebService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:3000")
public class WebController {

    WebService webService;

    public WebController(WebService webService) {
        this.webService = webService;
    }

    @PostMapping("/chat")
    public ResponseEntity<?> handleChat(@RequestBody Map<String, Object> request) {

        String userMessage = request.get("message").toString();

        Map<String,String> response = webService.chat(userMessage);

        return ResponseEntity.ok(response);
    }

}
