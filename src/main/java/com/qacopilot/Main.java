package com.qacopilot;

import com.qacopilot.llm.OllamaClient;

public class Main {
    public static void main(String[] args) {
        if (args.length > 0) {
            String prompt = String.join(" ", args);
            String answer = new OllamaClient().ask(prompt);
            System.out.println(answer);
            return;
        }

        System.out.println("QA Copilot started");
    }
}
