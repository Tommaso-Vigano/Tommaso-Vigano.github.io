---
layout: post
title: "How Does Zero-Copy Improve Apache Kafka Performance?"
date: 2026-10-14
description: "Kafka can send log data to consumers without copying it through application memory. Here is how zero-copy reduces the work involved in moving bytes from disk to the network."
categories: [technology, data, kafka]
image: /assets/images/zero-copy.png
---

While studying Apache Kafka, I came across zero-copy, one of the ingredients behind its efficient I/O. The idea is to avoid copying message data into the broker’s application memory when all the broker needs to do is send those bytes to a consumer. But which copies can we avoid, and why does that matter?

## Without zero-copy

Before looking at the optimization, let’s follow the conventional path. The diagram separates user space, where Kafka runs, from kernel space, where the operating system manages file and network I/O. Both the application buffer and the OS page cache are in RAM.

When a producer sends records, the broker receives them and appends them to its log. These writes initially reach the OS page cache and are flushed to disk later.

Now suppose a consumer requests records that are not in the cache. The conventional file-to-socket path involves four data transfers:

1. Disk to page cache: the OS loads the requested bytes into kernel memory.
2. Page cache to application buffer: Kafka reads the bytes into user space.
3. Application buffer to socket buffer: Kafka asks the OS to send those bytes.
4. Socket buffer to network interface: the data is transferred for transmission to the consumer.

In the simplified `read()` plus `write()` model, this involves four transfers and four user/kernel mode transitions: entering and returning from each system call. These mode transitions are often called context switches in explanations of zero-copy; they do not necessarily mean switching to another process.

The expensive round trip is from kernel memory into Kafka’s application buffer and back into kernel memory. Kafka is forwarding the same bytes, so copying them through user space adds work without changing the message.

![Kafka broker write and conventional read paths without zero-copy](/assets/images/without-zero-copy.png)

*Conventional file-to-network path without zero-copy*

## The zero-copy path

On Linux, `sendfile()` lets an application request a file-to-socket transfer without first reading the payload into a user-space buffer. It is a system call, rather than a shell command. Kafka can use this path to send log data to consumers over plaintext connections.


![Kafka broker read path using zero-copy to bypass the application buffer](/assets/images/zero-copy.png)

*Optimized path that bypasses the application buffer*


In that optimized model, a cache miss requires two hardware transfers: disk to RAM and RAM to the network interface. A single `sendfile()` call also reduces the simplified model to two user/kernel mode transitions. “Zero-copy” refers to avoiding CPU copies of the payload along this path;

If the records are already in the page cache, the disk read is skipped (as before). Consumers that keep up with producers can therefore fetch recently written data directly from RAM. Avoiding repeated copies is especially useful when several consumers request the same records.

There is a practical limit: Kafka’s standard TLS/SSL transport encrypts data in user space and does not use this `sendfile()` path. The diagrams describe the optimized plaintext case.

## What happens under the hood

In Java, `FileChannel.transferTo()` expresses a file-to-channel transfer. On a supported Linux file-to-socket path, it can use `sendfile()`. The following illustrative snippet shows the idea without reproducing Kafka’s internal implementation:

```java
// fileChannel is a FileChannel; socketChannel is a SocketChannel.
// position is a byte offset in the file, not a Kafka record offset.
long transferred = fileChannel.transferTo(
    position, remaining, socketChannel
);
position += transferred;
remaining -= transferred;
```

A call can transfer fewer bytes than requested, including zero. The caller must retain the updated position and remaining byte count and resume as needed. If a non-blocking socket cannot accept more data, it should wait until the socket is writable before retrying, rather than repeatedly spinning

A consumer fetch follows this simplified sequence:

1. The consumer requests records from a partition, starting at a record offset such as 1000.
2. The broker locates the log segment and determines the byte range to return. A Kafka record offset is not a byte position in the file.
3. For an eligible plaintext transfer, the broker uses a file-to-socket transfer rather than reading the payload into an application buffer.
4. The OS loads any missing file pages from disk. Cached pages need no disk read.
5. The networking stack and network interface transmit the bytes to the consumer.

What interested me about this design is how much of the optimization comes from removing unnecessary work. The broker can serve records efficiently by reusing the OS page cache and letting the kernel move the stored bytes toward the network.

---

## Sources and links

- [Apache Kafka design documentation — efficiency and page cache](https://kafka.apache.org/40/design/design/)
- [IBM Developer — Java ZeroCopy I/O optimization](https://developer.ibm.com/articles/j-zerocopy/)
- [Oracle Java documentation — FileChannel.transferTo](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html)
- [ByteByteGo — Why is Kafka fast](https://blog.bytebytego.com/p/why-is-kafka-fast)
- [2 Minute Streaming — Apache Kafka zero-copy](https://blog.2minutestreaming.com/p/apache-kafka-zero-copy-operating-system-optimization)
- [Krishna Konar — Kafka zero-copy architecture](https://krishnakonar12.medium.com/kafkas-zero-copy-the-architecture-behind-lightning-fast-message-delivery-7ffde4aadb7e)
