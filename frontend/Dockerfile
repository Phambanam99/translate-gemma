# ── Stage 1: Build React app ───────────────────────────────────────────
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline --no-audit --no-fund
COPY . .
RUN npm run build

# ── Stage 2: Nginx serves static + reverse-proxies /api → backend ─────
FROM nginx:1.27-alpine

# Copy built frontend
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config: SPA + reverse proxy to backend
RUN cat > /etc/nginx/conf.d/default.conf <<'NGINX'
upstream backend {
    server backend:8028;
    keepalive 32;
}

server {
    listen 80;
    server_name _;

    # ── Gzip ──
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 4;
    gzip_types text/plain text/css text/javascript application/json application/javascript image/svg+xml;
    gzip_min_length 256;

    # ── Upload limit (CSV files can be large) ──
    client_max_body_size 200M;

    # ── Reverse proxy: /api → backend ──
    location /api/ {
        proxy_pass         http://backend/api/;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Connection        "";
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering    off;
    }

    # ── Static assets (hashed filenames → long cache) ──
    location /assets/ {
        expires 365d;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # ── SPA fallback ──
    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache";
    }
}
NGINX

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO /dev/null http://localhost/ || exit 1

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
