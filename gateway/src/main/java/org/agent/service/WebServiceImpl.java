package org.agent.service;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.net.URI;
import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class WebServiceImpl implements WebService {

    RestTemplate restTemplate = new RestTemplate();

    @Override
    public Map<String, Object> chat(String message) {
        Map<String,String> request = new HashMap<>();
        request.put("input", message);

        Map<String, Object> result = new HashMap<>();
        result = restTemplate.postForObject(URI.create("http://localhost:8000/chat"), request,Map.class);
        result.put("message", message);

        return result;
    }
}
