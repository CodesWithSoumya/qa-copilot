package com.qacopilot.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

class OllamaClientTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void askReturnsResponseTextWhenServerReplies200() throws Exception {
        HttpServer server = createServer(exchange -> {
            String responseBody = "{\"response\":\"hello from ollama\"}";
            sendJson(exchange, 200, responseBody);
        });

        try {
            OllamaClient client = new OllamaClient("http://localhost:" + server.getAddress().getPort(), "llama3", Duration.ofSeconds(5));
            assertEquals("hello from ollama", client.ask("hi"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void askJsonSendsFormatJsonInRequestBody() throws Exception {
        HttpServer server = createServer(exchange -> {
            String requestBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            JsonNode json = OBJECT_MAPPER.readTree(requestBody);
            assertEquals("json", json.get("format").asText());
            sendJson(exchange, 200, "{\"response\":\"json reply\"}");
        });

        try {
            OllamaClient client = new OllamaClient("http://localhost:" + server.getAddress().getPort(), "llama3", Duration.ofSeconds(5));
            assertEquals("json reply", client.askJson("hi"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void non200ReplyThrowsOllamaExceptionContainingStatusCode() throws Exception {
        HttpServer server = createServer(exchange -> sendJson(exchange, 500, "{\"error\":\"bad\"}"));

        try {
            OllamaClient client = new OllamaClient("http://localhost:" + server.getAddress().getPort(), "llama3", Duration.ofSeconds(5));
            OllamaException ex = assertThrows(OllamaException.class, () -> client.ask("hi"));
            assertTrue(ex.getMessage().contains("500"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void invalidJsonReplyThrowsInvalidJsonMessage() throws Exception {
        HttpServer server = createServer(exchange -> sendJson(exchange, 200, "not json at all"));

        try {
            OllamaClient client = new OllamaClient("http://localhost:" + server.getAddress().getPort(), "llama3", Duration.ofSeconds(5));
            OllamaException ex = assertThrows(OllamaException.class, () -> client.ask("hi"));
            assertTrue(ex.getMessage().contains("invalid JSON"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void timeoutReplyThrowsDidNotAnswerMessage() throws Exception {
        HttpServer server = createServer(exchange -> {
            try {
                Thread.sleep(1500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            sendJson(exchange, 200, "{\"response\":\"too late\"}");
        });

        try {
            OllamaClient client = new OllamaClient("http://localhost:" + server.getAddress().getPort(), "llama3", Duration.ofSeconds(1));
            OllamaException ex = assertThrows(OllamaException.class, () -> client.ask("hi"));
            assertTrue(ex.getMessage().contains("did not answer"));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void unreachableServerThrowsCannotReachOllamaMessage() {
        int port = 65534;
        OllamaClient client = new OllamaClient("http://localhost:" + port, "llama3", Duration.ofSeconds(2));

        OllamaException ex = assertThrows(OllamaException.class, () -> client.ask("hi"));
        assertTrue(ex.getMessage().contains("Cannot reach Ollama at http://localhost:" + port));
    }

    private static HttpServer createServer(HttpHandler handler) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/api/generate", handler);
        server.start();
        return server;
    }

    private static void sendJson(HttpExchange exchange, int statusCode, String body) throws IOException {
        byte[] responseBytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(statusCode, responseBytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(responseBytes);
        }
    }
}
