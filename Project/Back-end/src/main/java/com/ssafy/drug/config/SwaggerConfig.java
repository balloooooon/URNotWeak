package com.ssafy.drug.config;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Info;
import lombok.RequiredArgsConstructor;
//import org.springdoc.core.GroupedOpenApi;
import org.springdoc.core.models.GroupedOpenApi;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@OpenAPIDefinition(
        info = @Info(title = "약해지지마 API 명세서",
                description = "마약에 대한 호기심을 방지하고 예방하는 API 명세서",
                version = "v1"))
@RequiredArgsConstructor
@Configuration
public class SwaggerConfig {
//    @Bean
//    public GroupedOpenApi chatOpenApi() {
//        String[] paths = {"/com/ssafy/drug/controller"};
//
//        return GroupedOpenApi.builder()
//                .group("약해지지마 API v1")
//                .pathsToMatch(paths)
//                .build();
//    }
}
