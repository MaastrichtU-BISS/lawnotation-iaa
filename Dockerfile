# Build stage
FROM golang:1.24-alpine AS build
WORKDIR /src
# The tool is a module now: the CLI entry point at the root plus the iaa
# package that holds everything else. Still no third-party dependencies, so
# there is nothing to download.
COPY go.mod ./
COPY main.go ./
COPY iaa/ ./iaa/
RUN CGO_ENABLED=0 go build -o /iaa .

# Runtime stage
FROM scratch
COPY --from=build /iaa /iaa
EXPOSE 8080
ENTRYPOINT ["/iaa", "--serve"]
