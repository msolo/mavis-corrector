# MavisCorrector

This is a small python component packaged as a MacOS exectuable that serves text correction results via JSON over HTTP or via a local unix domain socket.

MavisCorrector advertises its HTTP service as `_mavis-corrector._tcp` on the local domain so that the iOS version of Mavis can use this for providing text corrections. It will continue serving requests even while the host machine is sleeping, as long as it is connected to AC power.

# Building

This application is generally embedded as a dependency inside the MacOS version of Mavis, but must be built separately using the Python toolchain.

The main external dependency is `brew` - once that is available, the `build.sh` script will create download the model, compile the executable and correctly bless the bundle for MacOS.

```
./build.sh release
```

# Running Manually

```
 ./derived.noindex/dist/MavisCorrector.plugin/Contents/MacOS/MavisCorrector --http-service --http-port=62044
```

# Testing The HTTP Server

## One Ping Only
```
> curl localhost:62044/ping
{"error": null, "return": "pong", "id": 1750376689868900096}
```

## A Simple Sample Request
```
> curl -s localhost:62044/correct?text=something+heinours | jq .
{
  "error": null,
  "return": [
    "something he in ours",
    "something in ours",
    "something heinous",
    "something here in ours",
    "something heinours"
  ],
  "id": 1750376787276856064
}
```

# Release Process

> NOTE: Due to how py2app works internally, app bundles are not backward compatible across major releases of MacOS. `build-remote.sh` explains how to build for an older relase using a local VM.

 * Update the CFBundleVersion key in setup-app.py and run `./build.sh release` when a change is made.
 * `git tag release-0.1.2`


# Future

This is mainly a rapid proof of concept as opposed to a robust application. Requests are neither signed nor encrypted.

