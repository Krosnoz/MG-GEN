import re

from config.config_logomotion import Animation_CONF, AI_CONF

class PopupAnimation:
    @staticmethod
    def get_prompt(prompt_lang):
        prompt_en = """
Given image 1 shows the advertisement thumbnail we want to animate.
The HTML below represents the image in HTML format:
```
[IMAGE_HTML]
```

Reference context (recent best practices and API details):
```
[RAG_CONTEXT]
```

Please generate an animation idea to make the decomposed image layers and text appear on screen, taking into account the layout and the content of the advertisement above.
The IDs and contents of the layers we want to animate are as follows:
```
[LAYER_INFO]
```

Optionally consider the following (they may be empty):
- Previous idea: [ANIMATION_IDEA]
- Previous script: [ANIMATION_SCRIPT]
- Edit request: [ANIMATION_IDEA_EDIT]

Then, generate an animation script using anime.js v4 to implement these animations. The animation code should be written as follows:
- Use a single `anime.timeline` and add animations using `.add`
- Apply one animation per layer
- Coordinate movements of animations within the same group
- Configure timeline with `loop=false` and `autoplay=true`

Output format (enclose content in the specified tags):
```
## Animation idea
<idea>
- **{Animation group 1 name}:**
    1. {Description of animation}
    2. {Description of animation}
    ...

- **{Animation group 2 name}:**
    1. {Description of animation}
    2. {Description of animation}
    ...
...
</idea>

## Animation code
<script>
const tl = anime.timeline({ loop: false, autoplay: true });

// Animation group 1
tl.add({
    ...
}).add({
    ...
});

// Animation group 2
tl.add({
    ...
}).add({
    ...
});

// Add more groups if needed
...
</script>
```
        """

        # Force English prompt regardless of input language to standardize behavior
        return prompt_en

    @staticmethod
    def script_postprocess(script):
        match = [m.group() for m in re.finditer(r'anime.timeline\(\{(.*?)\}\)\;', script, re.DOTALL)]
        if match and Animation_CONF.options.loop is not None:
            timeline = match[-1] 
            loop_count = Animation_CONF.options.loop.count
            script = script.replace(timeline, "anime.timeline({loop: " + str(loop_count*2) + ", autoplay: true, direction: 'alternate', complete: function() {setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 1000);}});")
            return script
        
        elif match:
            timeline = match[-1]
            script = script.replace(timeline, "anime.timeline({loop: false, autoplay: true, complete: function() {setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 1000);}});")
            return script

        else:
            script = "/* Script has been deleted because the format is not correct. */ \n setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 5000);"  
            return script