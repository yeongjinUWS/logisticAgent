package org.agent.service;

import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

public interface WebService {
    Map<String, Object> chat(String message);

    Map<String, Object> uploadExcel(MultipartFile file);

    Map<String, Object> learningModel(MultipartFile file, List<String> columns);

    Map<String, Object> getModels();
}
