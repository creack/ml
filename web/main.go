// Copyright 2015 The Gorilla WebSocket Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"encoding/base64"

	"github.com/gorilla/websocket"
)

var addr = flag.String("addr", "localhost:8080", "http service address")

var upgrader = websocket.Upgrader{} // use default options

var tpl = `set terminal png transparent nocrop enhanced size 450,320 font "arial,8"
set key inside left top vertical Right noreverse enhanced autotitle box lt black linewidth 1.000 dashtype solid
set samples 50, 50
set title "Simple Plots"
set title  font ",20" norotate
plot [-10:10] cos(atan(x+{{.}}))
`

func echo(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Print("upgrade:", err)
		return
	}
	defer c.Close()
	i := 0
	for {
		time.Sleep(500 * time.Millisecond)
		// mt, message, err := c.ReadMessage()
		// if err != nil {
		// 	log.Println("read:", err)
		// 	break
		// }
		// log.Printf("recv: %s", message)

		cmd := exec.Command("gnuplot")
		cmd.Stdin = strings.NewReader(strings.Replace(tpl, "{{.}}", fmt.Sprint(i), -1))
		buf, err := cmd.CombinedOutput()
		if err != nil {
			log.Println("gnuplot:", err)
			break
		}

		// TODO: encode to byte to avoid triple copy.
		message := base64.StdEncoding.EncodeToString(buf)

		if err := c.WriteMessage(websocket.TextMessage, []byte(message)); err != nil {
			log.Println("write:", err)
			break
		}
		i++
	}
}

func home(w http.ResponseWriter, r *http.Request) {
	homeTemplate.Execute(w, "ws://"+r.Host+"/echo")
}

func main() {
	flag.Parse()
	log.SetFlags(0)
	http.HandleFunc("/echo", echo)
	http.HandleFunc("/", home)
	log.Fatal(http.ListenAndServe(*addr, nil))
}

var homeTemplate = template.Must(template.New("").Parse(`
<!DOCTYPE html>
<head>
<meta charset="utf-8">
<script>
window.addEventListener("load", function(evt) {
    var output = document.getElementById("output");
    var input = document.getElementById("input");
    var stream = document.getElementById("stream");
    var ws;
    var print = function(message) {
        var d = document.createElement("div");
        d.innerHTML = message;
        output.appendChild(d);
    };
    document.getElementById("open").onclick = function(evt) {
        if (ws) {
            return false;
        }
        ws = new WebSocket("{{.}}");
        ws.onopen = function(evt) {
            print("OPEN");
        }
        ws.onclose = function(evt) {
            print("CLOSE");
            ws = null;
        }
        ws.onmessage = function(evt) {
        var dd = document.createElement("div");
        dd.innerHTML = evt.data;
        stream.setAttribute('src', 'data:image/png;base64,' + evt.data);

        }
        ws.onerror = function(evt) {
            print("ERROR: " + evt.data);
        }
        return false;
    };
    document.getElementById("send").onclick = function(evt) {
        if (!ws) {
            return false;
        }
        print("SEND: " + input.value);
        ws.send(input.value);
        return false;
    };
    document.getElementById("close").onclick = function(evt) {
        if (!ws) {
            return false;
        }
        ws.close();
        return false;
    };
});
</script>
</head>
<body>
<table>
<tr><td width="20%">
<p>Click "Open" to create a connection to the server,
"Send" to send a message to the server and "Close" to close the connection.
You can change the message and send multiple times.
<p>
<form>
<button id="open">Open</button>
<button id="close">Close</button>
<p><input id="input" type="text" value="Hello world!">
<button id="send">Send</button>
</form>
</td><td valign="top" width="50%">
<img id="stream" />
</td><td>
<div id="output"></div>
</td></tr></table>
</body>
</html>
`))
