# Build stage
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY main.go iaa.go server.go ./
RUN CGO_ENABLED=0 go build -o /iaa main.go iaa.go server.go

# Runtime stage
FROM scratch
COPY --from=build /iaa /iaa
EXPOSE 8080
ENTRYPOINT ["/iaa", "--serve"]
