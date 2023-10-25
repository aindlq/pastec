Pastec
======

Introduction
------------

### Presentation

[Pastec](http://www.pastec.io) is an open source index and search engine for image recognition based on [OpenCV](http://www.opencv.org/). It can recognize flat objects such as covers, packaged goods or artworks. It has, however, not been designed to recognize faces, 3D objects, barcodes, or QR codes.

Pastec can be, for example, used to recognize DVD covers in a mobile app or detect near duplicate images in a big database.

Pastec does not store the pixels of the images in its database. It stores a signature of each image thanks to the technique of [visual words](http://en.wikipedia.org/wiki/Visual_Word).

Pastec offers a tiny HTTP API using JSON to add, remove, and search for images in the index.

### Intellectual property

Pastec is developed by [Visualink](http://www.visualink.io) and licenced under the [GNU LGPL v3.0](http://www.gnu.org/licenses/lgpl.html).
It is based on the free packages of [OpenCV](http://www.opencv.org/) that are available for commercial purposes; you should therefore be free to use Pastec without paying for any patent license.

More precisely, Pastec uses the [patent-free ORB descriptor](https://www.willowgarage.com/sites/default/files/orb_final.pdf) and not the well-known SIFT and SURF descriptors that are patented.

Setup
-----



## Compilation ###
 Dependancies To be compiled, Pastec requires [OpenCV 3.X](https://web.archive.org/web/20201207220940/http://www.opencv.org/) and [libmicrohttpd](https://web.archive.org/web/20201207220940/http://www.gnu.org/software/libmicrohttpd/) and libcurl. On **Ubuntu 18.04**, those package can be installed using the following command:

<pre data-language="shell">sudo apt-get install libopencv-dev libmicrohttpd-dev
</pre>

If you are using another distribution or operating system, you may have to compile them yourself. ### Building Pastec uses cmake as build system. You also need Git to get the source code. On Ubuntu, they can be installed using the following command:

<pre data-language="shell">sudo apt-get install cmake git
</pre>

To compile Pastec, first get the sources with the following command:

<pre data-language="shell">git clone https://github.com/Visu4link/pastec.git
cd pastec
</pre>

Then create a compilation folder:

<pre data-language="shell">mkdir build
</pre>

Go to this subdirectory and run cmake:

<pre data-language="shell">cd build
cmake ../
</pre>

Finally, run make to compile Pastec:

<pre data-language="shell">make
</pre>

## Running

 To start Pastec, just run the **pastec** executable. It takes as mandatory argument the path to a file containing a list of ORB visual words. For now,  use the file visualWordsORB.dat. Next Pastec releases will contain tools that will allow you to generate your own list of visual words.

<pre data-language="shell">./pastec visualWordsORB.dat
</pre>

The default port pastec listens for the REST API is **4212**. You can set an other port with the **-p** argument. You can also give a path to an index file to load with the **-i** argument.</div>

API
-----


#### HTTP API

Pastec can be controlled using a simple HTTP API. By default, it listens to the 4212 port but you can change this using the -p argument.

Pastec answers are always formatted in JSON. They contains a mandatory type field that describes the result obtained or an error. Each image has an associated id that is a 32 bit unsigned integer. This id establishes the link in the index between the images and their signatures.

All the uploaded images must have their **dimensions** above **150 pixels**. If one of the image dimension exceeds 1000 pixels, the image is resized so that the maximum dimension is set to 1000 pixels and the original aspect ratio is kept.

Here is a detailed list of the API calls:


### Adding an image to the index

This call allows to add the signature of an image in the index to make it available for searching. You need to provide the compressed binary data of the image and an id to identify it.

*   **Path:** /index/images/<image id>
*   **HTTP method:** POST
*   **Data:** the binary data of the image to add compressed in JPEG **or** a JSON containing the URL of the image in the "url" field.
*   **Answer type:** "IMAGE_ADDED"
*   **Possible error types:** "IMAGE_NOT_DECODED", "IMAGE_SIZE_TOO_BIG", "IMAGE_SIZE_TOO_SMALL", "IMAGE_DOWNLOADER_HTTP_ERROR" with the HTTP status code in the "image_downloader_http_response_code" field.
*   **Example:**
    *   Command line with image data:

        <pre data-language="shell">curl -X POST --data-binary @/home/test/img/1.jpg http://localhost:4212/index/images/23
        </pre>

    *   Answer:

        <pre data-language="json">{
           "image_id" : 23,
           "type" : "IMAGE_ADDED"
        }
        </pre>

    *   Command line with an image URL:

        <pre data-language="shell">curl -X POST -d '{"url":"http://www.mydomain.com/path/to/my/image.jpg"}' http://localhost:4212/index/images/26
        </pre>

    *   Answer:

        <pre data-language="json">{
           "image_id" : 26,
           "type" : "IMAGE_ADDED"
        }
        </pre>

### Removing an image from the index

This call removes the signature of an image in the index thanks to its id. Be careful to not call often this method if your index is big because it is currently very slow.

*   **Path:** /index/images/<image id>
*   **HTTP method:** DELETE
*   **Answer type:** "IMAGE_REMOVED"
*   **Possible error type:** "IMAGE_NOT_FOUND"
*   **Example:**
    *   Command line:

        <pre data-language="shell">curl -X DELETE http://localhost:4212/index/images/23
        </pre>

    *   Answer:

        <pre data-language="json">{
           "image_id" : 23,
           "type" : "IMAGE_REMOVED"
        }
        </pre>

### Search request

This call performs a search in the index thanks to a request image. It returns the id of the matched images from the most to the least relevant ones.

Request JPEG images with a size approximately equal to 450x340 pixels and a 75% quality are usally enough to achieve a good matching. Their small size allows to quickly send them over a mobile network.

*   **Path:** /index/searcher
*   **HTTP method:** POST
*   **Data:** the binary data of the request image compressed in JPEG **or** a JSON containing the URL of the image in the "url" field.
*   **Answer:** "SEARCH_RESULTS" as type field and a list of the the matched image ids from the most to the least relevant one in the "image_ids" field
*   **Possible error types:** "IMAGE_NOT_DECODED", "IMAGE_SIZE_TOO_BIG", "IMAGE_SIZE_TOO_SMALL"
*   **Example:**
    *   Command line with image data:

        <pre data-language="shell">curl -X POST --data-binary @/home/test/img/request.jpg http://localhost:4212/index/searcher
        </pre>

    *   Answer:

        <pre data-language="json">{
           "image_ids" : [ 2, 5, 43 ],
           "type" : "SEARCH_RESULTS"
        }
        </pre>

    *   Command line with an image URL:

        <pre data-language="shell">curl -X POST -d '{"url":"http://www.mydomain.com/path/to/my/image.jpg"}' http://localhost:4212/index/searcher
        </pre>

    *   Answer:

        <pre data-language="json">{
           "image_ids" : [ 8, 12, 73 ],
           "type" : "SEARCH_RESULTS"
        }
        </pre>

### Clear an index

This call erases all the data currently contained in the index.

*   **Path:** /index/io
*   **HTTP method:** POST
*   **Answer type:** "INDEX_CLEARED"
*   **Possible error types:** -
*   **Example:**
    *   Command line:

        <pre data-language="shell">curl -X POST -d '{"type":"CLEAR"}' http://127.0.0.1:4212/index/io
        </pre>

    *   Answer:

        <pre data-language="json">{
           "type" : "INDEX_CLEARED"
        }
        </pre>

### Load an index

This call loads the index data in a provided path.

*   **Path:** /index/io
*   **HTTP method:** POST
*   **Data:** a json with a type field of value "LOAD" and a "index_path" field that set the path where to read the index.
*   **Answer type:** "INDEX_LOADED"
*   **Possible error types:** "INDEX_NOT_FOUND"
*   **Example:**
    *   Command line:

        <pre data-language="shell">curl -X POST -d '{"type":"LOAD", "index_path":"test.dat"}' http://127.0.0.1:4212/index/io
        </pre>

    *   Answer:

        <pre data-language="json">{
           "type" : "INDEX_LOADED"
        }
        </pre>

### Save an index

This call saves the index data in a specified path.

*   **Path:** /index/io
*   **HTTP method:** POST
*   **Data:** a json with a type field of value "WRITE" and a "index_path" field that set the path where to write the index
*   **Answer type:** "INDEX_WRITTEN"
*   **Possible error types:** "INDEX_NOT_WRITTEN"
*   **Example:**
    *   Command line:

        <pre data-language="shell">curl -X POST -d '{"type":"WRITE", "index_path":"test.dat"}' http://127.0.0.1:4212/index/io
        </pre>

    *   Answer:

        <pre data-language="json">{
           "type" : "INDEX_WRITTEN"
        }
        </pre>

### Ping Pastec

This call sends a simple PING command to pastec that answers with a PONG.

*   **Path:** /
*   **HTTP method:** POST
*   **Data:** a json with a "type" field of value "PONG"
*   **Answer type:** "PONG"
*   **Possible error types:** -
*   **Example:**
    *   Command line:

        <pre data-language="shell">curl -X POST -d '{"type":"PING"}' http://localhost:4212/
        </pre>

    *   Answer:

        <pre data-language="json">{
           "type" : "PONG"
        }
        </pre>

## Python API

In the python subdirectory of the source directory, you will also find a tiny python API that is actually just a wrapper of the HTTP API. We encourage you to read the small source to code to understand it.

