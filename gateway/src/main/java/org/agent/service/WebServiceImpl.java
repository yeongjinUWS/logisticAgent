package org.agent.service;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.agent.service.weather.WeatherAlert;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import com.google.gson.JsonObject;
@Slf4j
@Service
@RequiredArgsConstructor
public class WebServiceImpl implements WebService {

    RestTemplate restTemplate = new RestTemplate();

    @Override
    public Map<String, Object> chat(String message) {
        Map<String,Object> request = new HashMap<>();

        JsonObject  object = new JsonObject();
        try {
            object = WeatherAlert.getWeather(String.valueOf(37),String.valueOf(127));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.out.println(object);
        JsonArray array = new JsonArray();
        array = object.getAsJsonObject("response").getAsJsonObject("body").getAsJsonObject("items").getAsJsonArray("item");
        System.out.println(array.toString());
        Map<String,String> reqWeather = new HashMap<>();
        for(int i=0;i<array.size();i++){
            JsonElement element = array.get(i);

            reqWeather.put(element.getAsJsonObject().get("category").getAsString(), element.getAsJsonObject().get("obsrValue").getAsString());
        }
        Map<String,Object> weather = new HashMap<>();
        weather.put("T1H", Double.parseDouble(reqWeather.get("T1H")));
        weather.put("RN1", Double.parseDouble(reqWeather.get("RN1")));
        weather.put("PTY", Integer.parseInt(reqWeather.get("PTY")));
        weather.put("REH", Double.parseDouble(reqWeather.get("REH")));
        weather.put("WSD", Double.parseDouble(reqWeather.get("WSD")));
        request.put("input", message);
        request.put("weather", weather);
        Map<String, Object> result = new HashMap<>();
        result = restTemplate.postForObject(URI.create("http://localhost:8000/chat"), request,Map.class);
        result.put("message", message);

        return result;
    }
}
