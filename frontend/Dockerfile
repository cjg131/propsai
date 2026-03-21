FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

ENV NEXT_PUBLIC_API_URL=http://localhost:8000
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
