package org.agent.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.agent.service.WebService;
import org.springframework.boot.configurationprocessor.json.JSONArray;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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

        Map<String,Object> response = webService.chat(userMessage);

        return ResponseEntity.ok(response);
    }

    @PostMapping("/upload")
    public ResponseEntity<?> handleUpload(@RequestParam("file") MultipartFile file) {
        Map<String,Object> response = new HashMap<>();
        return ResponseEntity.ok(webService.uploadExcel(file));
    }

    @PostMapping("/learning")
    public ResponseEntity<?> handleLearning(@RequestParam("file") MultipartFile file, @RequestParam("column") List<String> columnsJson) {
        return ResponseEntity.ok(webService.learningModel(file,columnsJson));
    }

    @PostMapping("/getModels")
    public ResponseEntity<?> handleGetModels() {
        return ResponseEntity.ok(webService.getModels());
    }
}
