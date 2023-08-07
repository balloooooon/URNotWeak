# 빌드된 산출물을 실행시키기 위해 필요한 serve 모듈
RUN npm install -g serve

# 작업 공간
RUN mkdir /app
WORKDIR /app

# 빌드된 산출물 도커 이미지로 복사
RUN mkdir ./build
COPY ./build ./build

# 실행 명령어
ENTRYPOINT ["serve", "-s", "build"]

# # nginx 이미지를 사용합니다. 뒤에 tag가 없으면 latest 를 사용.
# # FROM nginx

# # root 에 app 폴더를 생성
# RUN mkdir /app

# # work dir 고정
# WORKDIR /app

# # work dir 에 build 폴더 생성 /app/build
# RUN mkdir ./build

# # host pc의 현재경로의 build 폴더를 workdir 의 build 폴더로 복사
# ADD ./build ./build

# # # nginx 의 default.conf 를 삭제
# RUN rm /etc/nginx/conf.d/default.conf

# # # host pc 의 nginx.conf 를 아래 경로에 복사
# # COPY ./nginx.conf /etc/nginx/conf.d

# # 3000 포트 오픈
# EXPOSE 3000

# # container 실행 시 자동으로 실행할 command. nginx 시작함
# CMD ["nginx", "-g", "daemon off;"]

