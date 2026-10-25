package com.qacopilot.llm;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Client for talking to a local Ollama server.
 */
public class OllamaClient {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final String DEFAULT_MODEL = "qwen2.5-coder:latest";

    private final String baseUrl;
    private final String model;
    private final HttpClient httpClient;
    private final Duration timeout;

    /**
     * Creates a client for the given Ollama endpoint and model.
     *
     * @param baseUrl the Ollama base URL, such as http://localhost:11434
     * @param model the Ollama model name to use
     * @param timeout the request timeout
     */
    public OllamaClient(String baseUrl, String model, Duration timeout) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.model = model;
        this.timeout = timeout;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .build();
    }

    /**
     * Creates a client using the default local Ollama configuration.
     */
    public OllamaClient() {
        this("http://localhost:11434", resolveDefaultModel(System.getenv("OLLAMA_MODEL")), Duration.ofSeconds(120));
    }

    static String resolveDefaultModel(String envValue) {
        if (envValue == null || envValue.trim().isEmpty()) {
            return DEFAULT_MODEL;
        }
        return envValue.trim();
    }

    /**
     * Sends a prompt to Ollama and returns the generated text response.
     *
     * @param prompt the prompt to send
     * @return the generated response text
     */
    public String ask(String prompt) {
        return askInternal(prompt, false);
    }

    /**
     * Sends a prompt to Ollama and requests a JSON-formatted response.
     *
     * @param prompt the prompt to send
     * @return the generated response text
     */
    public String askJson(String prompt) {
        return askInternal(prompt, true);
    }

    private String askInternal(String prompt, boolean jsonResponse) {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("model", model);
            body.put("prompt", prompt);
            body.put("stream", false);
            if (jsonResponse) {
                body.put("format", "json");
            }

            String requestBody = OBJECT_MAPPER.writeValueAsString(body);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/generate"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                    .timeout(timeout)
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() != 200) {
                throw new OllamaException("HTTP " + response.statusCode() + ": " + response.body());
            }

            Map<String, Object> payload = OBJECT_MAPPER.readValue(response.body(), new TypeReference<Map<String, Object>>() {
            });
            Object responseText = payload.get("response");
            if (responseText == null) {
                throw new OllamaException("Ollama response is missing the 'response' field.");
            }
            return responseText.toString();
        } catch (HttpTimeoutException e) {
            throw new OllamaException("Ollama did not answer within " + timeout.getSeconds() + " seconds. Try a smaller model or a longer timeout.", e);
        } catch (JsonProcessingException e) {
            throw new OllamaException("Ollama returned an invalid JSON response.", e);
        } catch (IOException e) {
            throw new OllamaException("Cannot reach Ollama at " + baseUrl + ". Is it running? Start it with: ollama serve", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new OllamaException("The request to Ollama was interrupted.", e);
        } catch (IllegalArgumentException e) {
            throw new OllamaException("Invalid Ollama URL: " + baseUrl, e);
        }
    }
}
