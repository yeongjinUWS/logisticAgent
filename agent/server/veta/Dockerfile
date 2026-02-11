# 1. 베이스 이미지 선택
FROM python:3.11-slim

# 2. 필수 시스템 패키지 설치 (psycopg2 빌드용)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
# 아래의 requirements.txt 섹션 참고
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 모델 메타데이터 복사
COPY . .

# 6. 포트 노출 및 실행
EXPOSE 8004
CMD ["uvicorn", "vetaAgent:app", "--host", "0.0.0.0", "--port", "8004"]