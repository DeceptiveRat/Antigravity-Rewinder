## divide steps into categories
[[Divide steps]]

## Category 1. Prompt
Simplest. Just rewind to prompt. 

## Category 2. Tool Call/Result - A
Agent hung, waiting for any prompt to start it. 

To fix this, I added a proxy that removes the token `__AGY_KICKSTART__` from a prompt, so that I can prompt with it and not impact the output. 

![[kickstart_prompt.png]]
*proxy running* [^1]

It can be seen the conversation resumes. 

![[kickstart_no_proxy.png]]
*proxy not running*

Without the proxy running, the prompt is sent to the LLM, making it stop what it is doing. 

## Category 3. Tool Call/Result - B
Because tool calls for B are writing, they do not return results that influence the agent's further action. 

## Category 4. Tool Call/Result - C
Same with Category 2. We can change the result of the bash command run, thus influencing the agent's further action.

## Category 5. Tool Call/Result - D
Same with Category 2. But, because `read_url_content` saves the content to a file, then that file is read, it is better to manipulate that file itself, or the `view_file` to it. What can be manipulated for `read_url_content` is the message given when reading the URL, which usually states the contents can be found at a specific location.



[^1]: set up with:
	``` sh
	export AGY_KICKSTART_TOKEN="Stop doing everything and standby"
	mitmdump -s rewinding/interceptor.py -p 8080 > /tmp/mitmproxy.log 2>&1 &
	export HTTPS_PROXY="http://127.0.0.1:8080"
	export HTTP_PROXY="http://127.0.0.1:8080"
	export SSL_CERT_FILE="/home/sandbox/.mitmproxy/mitmproxy-ca-cert.pem"
	```
