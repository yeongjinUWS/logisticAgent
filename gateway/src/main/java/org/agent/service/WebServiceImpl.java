package org.agent.service;

import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class WebServiceImpl implements WebService {
    @Override
    public Map<String, String> chat(String message) {
        Map<String, String> response = new HashMap<>();
        response.put("message", message);
        response.put("result", "TEST");

        return response;
    }
}
