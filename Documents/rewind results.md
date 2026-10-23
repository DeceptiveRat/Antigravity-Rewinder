## simple rewind
When rewinding to prompt, it works seamlessly. It displays the previous response and waits for the next prompt. 

When rewinding to response, it displays the previous prompt, and nothing else, waits for next prompt as if previous prompt was interrupted. 

## complicated rewind
When rewound to tool response, it hangs and waits for previous prompt like with rewinding to response. 

When prompted to continue, it continues. 

![[rewind_to_tool_result_and_continue.png]]

But this risks changing how the tool is called, and with what parameters and such. 

