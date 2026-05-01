"""
Emotion Story Dataset Generator
 
This script generates stories depicting characters experiencing specific emotions,
following Anthropic's methodology from their emotion vectors paper.
 
Usage:
    python generate_emotion_stories.py --output_dir ./emotion_stories --all_emotions
    python generate_emotion_stories.py --output_dir ./emotion_stories --emotions_subset 80
    python generate_emotion_stories.py --resume  # Resume from checkpoint
    python generate_emotion_stories.py --stats ./emotion_stories/stories.jsonl
"""
 
import argparse
import json
import random
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import torch
from tqdm import tqdm

 
EMOTION_CATEGORIES = {
    "sadness": ["sad", "melancholy", "depressed", "heartbroken", "miserable", "grief-stricken", "lonely", "hurt", "unhappy", "dispirited", "gloomy"],
    "fear": ["afraid", "scared", "terrified", "frightened", "panicked", "anxious", "nervous", "alarmed", "paranoid", "horrified", "unnerved"],
    "anger": ["angry", "furious", "enraged", "irate", "hostile", "resentful", "bitter", "outraged", "mad", "irritated", "annoyed", "frustrated", "exasperated"],
    "joy": ["happy", "joyful", "elated", "euphoric", "ecstatic", "delighted", "cheerful", "jubilant", "thrilled", "excited", "blissful"],
    "surprise": ["surprised", "amazed", "astonished", "shocked", "dumbstruck", "bewildered", "mystified", "awestruck", "puzzled", "perplexed"],
    "disgust": ["disgusted", "contemptuous", "disdainful", "scornful", "offended", "insulted"],
    "calm": ["calm", "serene", "peaceful", "relaxed", "content", "at ease", "satisfied", "relieved"],
    "energy": ["energized", "invigorated", "vibrant", "stimulated", "alert", "vigilant", "eager", "enthusiastic", "exuberant"],
    "fatigue": ["tired", "weary", "sleepy", "sluggish", "listless", "worn out", "lazy", "bored"],
    "social_positive": ["loving", "compassionate", "empathetic", "sympathetic", "grateful", "thankful", "kind", "proud"],
    "social_negative": ["jealous", "envious", "resentful", "hateful", "vengeful", "vindictive", "spiteful", "hostile"],
    "self_conscious": ["embarrassed", "ashamed", "guilty", "humiliated", "mortified", "self-conscious", "regretful", "remorseful"],
    "uncertainty": ["unsettled", "uneasy", "restless", "troubled", "disturbed", "shaken", "rattled", "disoriented"],
    "determination": ["defiant", "stubborn", "obstinate", "valiant", "triumphant", "self-confident"],
}
 
 
def load_prompt_template(prompt_file: str = "prompts/story_prompt.txt") -> str:
    """Load the story generation prompt template from file."""
    with open(prompt_file, "r") as f:
        return f.read().strip()
 
 
def load_emotions(emotions_file: str = "prompts/emotions.txt") -> List[str]:
    """Load emotions from file (one per line)."""
    with open(emotions_file, "r") as f:
        emotions = [line.strip() for line in f if line.strip()]
    return emotions
 
 
def load_topics(topics_file: str = "prompts/topics.txt") -> List[str]:
    """Load topics from file (one per line)."""
    with open(topics_file, "r") as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics
 

def format_prompt(template: str, emotion: str, topic: str, n_stories: int = 3) -> str:
    """Format the prompt template with specific values."""
    return template.format(
        n_stories=n_stories,
        topic=topic,
        emotion=emotion,
    )
 
 
def parse_stories(response_text: str) -> List[str]:
    """Parse the generated response into individual stories."""
    stories = re.split(r'\[story\s*\d+\]', response_text, flags=re.IGNORECASE)
    # Filter empty strings and strip whitespace
    stories = [s.strip() for s in stories if s.strip()]
    return stories
 
 
def select_stratified_emotions(
    all_emotions: List[str], 
    n_emotions: int, 
    seed: int = 42
) -> List[str]:
    """Select a stratified subset of emotions covering different categories."""
    random.seed(seed)
    selected = []
    
    # First, try to get at least one from each category
    for category, category_emotions in EMOTION_CATEGORIES.items():
        available = [e for e in category_emotions if e in all_emotions and e not in selected]
        if available:
            selected.append(random.choice(available))
    
    # Fill remaining slots randomly from all emotions
    remaining = [e for e in all_emotions if e not in selected]
    random.shuffle(remaining)
    
    while len(selected) < n_emotions and remaining:
        selected.append(remaining.pop())
    
    return selected[:n_emotions]
 
 
class EmotionStoryGenerator:
    """Generator class for emotion stories using Apertus."""
    
    def __init__(
        self,
        model_name: str = "swiss-ai/Apertus-8B-Instruct-2509",
        device: str = "cuda",
        output_dir: str = "./emotion_stories",
        prompt_file: str = "prompts/story_prompt.txt",
        emotions_file: str = "prompts/emotions.txt",
        topics_file: str = "prompts/topics.txt",
        stories_per_prompt: int = 3,
        prompts_per_emotion: int = 3,
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.stories_per_prompt = stories_per_prompt
        self.prompts_per_emotion = prompts_per_emotion
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.prompt_template = load_prompt_template(prompt_file)
        self.all_emotions = load_emotions(emotions_file)
        self.all_topics = load_topics(topics_file)
        
        print(f"Loaded {len(self.all_emotions)} emotions from {emotions_file}")
        print(f"Loaded {len(self.all_topics)} topics from {topics_file}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for saving
        self.stories_file = self.output_dir / "stories.jsonl"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.metadata_file = self.output_dir / "metadata.json"
        
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model and tokenizer."""
        if self.model is not None:
            return
        
        print(f"Loading model: {self.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model loaded successfully")
    
    def generate_stories(self, emotion: str, topic: str) -> Dict:
        """Generate stories for a single emotion-topic pair."""
        prompt = format_prompt(
            self.prompt_template, 
            emotion, 
            topic, 
            self.stories_per_prompt
        )
        
        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and parse 
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        stories = parse_stories(response)
        
        return {
            "emotion": emotion,
            "topic": topic,
            "prompt": prompt,
            "raw_response": response,
            "stories": stories,
            "n_stories_parsed": len(stories),
            "timestamp": datetime.now().isoformat(),
        }
    
    def load_checkpoint(self) -> set:
        """Load checkpoint to get already completed emotion-topic pairs."""
        if not self.checkpoint_file.exists():
            return set()
        
        with open(self.checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        
        return set(tuple(pair) for pair in checkpoint.get("completed", []))
    
    def save_checkpoint(self, completed: set):
        """Save checkpoint with completed emotion-topic pairs."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "completed": list(completed),
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)
    
    def save_result(self, result: Dict):
        """Append a result to the stories file."""
        with open(self.stories_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    def save_metadata(self, emotions: List[str], topics: List[str]):
        """Save metadata about the generation run."""
        metadata = {
            "model_name": self.model_name,
            "n_emotions": len(emotions),
            "n_topics": len(topics),
            "stories_per_prompt": self.stories_per_prompt,
            "prompts_per_emotion": self.prompts_per_emotion,
            "total_expected_stories": len(emotions) * self.prompts_per_emotion * self.stories_per_prompt,
            "emotions": emotions,
            "topics_used": topics,
            "generation_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            "prompt_template": self.prompt_template,
            "started": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def run(
        self,
        emotions: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        resume: bool = True,
        seed: int = 42,
    ):
        """Run the full generation pipeline."""
        # Use defaults if not provided
        if emotions is None:
            emotions = self.all_emotions
        if topics is None:
            topics = self.all_topics
        
        # Set random seed for reproducibility
        random.seed(seed)
        self.load_model()
        self.save_metadata(emotions, topics)
        
        # Load checkpoint if resuming
        completed = self.load_checkpoint() if resume else set()
        print(f"Resuming from checkpoint: {len(completed)} pairs already completed")
        
        pairs = []
        for emotion in emotions:
            # Sample topics for this emotion
            sampled_topics = random.sample(topics, k=min(self.prompts_per_emotion, len(topics)))
            for topic in sampled_topics:
                pairs.append((emotion, topic))
        
        # Filter out completed pairs
        pairs_to_process = [p for p in pairs if p not in completed]
        print(f"Total pairs to process: {len(pairs_to_process)}")
        
        # Process with progress bar
        failed = []
        for emotion, topic in tqdm(pairs_to_process, desc="Generating stories"):
            try:
                result = self.generate_stories(emotion, topic)
                self.save_result(result)
                completed.add((emotion, topic))
                self.save_checkpoint(completed)
                
                # Log if parsing found unexpected number of stories
                if result["n_stories_parsed"] != self.stories_per_prompt:
                    print(f"\n  Warning: Expected {self.stories_per_prompt} stories, got {result['n_stories_parsed']} for {emotion}/{topic[:30]}...")
                    
            except Exception as e:
                print(f"\n  Error generating for {emotion}/{topic[:30]}...: {e}")
                failed.append((emotion, topic, str(e)))
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"Generation complete!")
        print(f"  Completed: {len(completed)} pairs")
        print(f"  Failed: {len(failed)} pairs")
        print(f"  Output: {self.stories_file}")
        
        if failed:
            failed_file = self.output_dir / "failed.json"
            with open(failed_file, "w") as f:
                json.dump(failed, f, indent=2)
            print(f"  Failed pairs saved to: {failed_file}")
 
 
def load_stories(stories_file: str) -> List[Dict]:
    """Load stories from JSONL file."""
    stories = []
    with open(stories_file, "r") as f:
        for line in f:
            stories.append(json.loads(line))
    return stories
 
 
def get_stories_by_emotion(stories_file: str) -> Dict[str, List[str]]:
    """Load stories grouped by emotion."""
    data = load_stories(stories_file)
    
    by_emotion = {}
    for item in data:
        emotion = item["emotion"]
        if emotion not in by_emotion:
            by_emotion[emotion] = []
        by_emotion[emotion].extend(item["stories"])
    
    return by_emotion
 
 
def print_statistics(stories_file: str):
    """Print statistics about the generated dataset."""
    data = load_stories(stories_file)
    
    total_stories = sum(item["n_stories_parsed"] for item in data)
    emotions = set(item["emotion"] for item in data)
    topics = set(item["topic"] for item in data)
    
    by_emotion = {}
    for item in data:
        emotion = item["emotion"]
        by_emotion[emotion] = by_emotion.get(emotion, 0) + item["n_stories_parsed"]
    
    print(f"\nDataset Statistics:")
    print(f"  Total prompts: {len(data)}")
    print(f"  Total stories: {total_stories}")
    print(f"  Unique emotions: {len(emotions)}")
    print(f"  Unique topics: {len(topics)}")
    print(f"  Avg stories per emotion: {total_stories / len(emotions):.1f}")
    print(f"\nStories per emotion (min/max):")
    print(f"  Min: {min(by_emotion.values())} ({min(by_emotion, key=by_emotion.get)})")
    print(f"  Max: {max(by_emotion.values())} ({max(by_emotion, key=by_emotion.get)})")
 
 
def main():
    parser = argparse.ArgumentParser(description="Generate emotion story dataset")
    
    parser.add_argument("--output_dir", type=str, default="./emotion_stories",
                        help="Output directory for stories")
    
    parser.add_argument("--prompt_file", type=str, default="prompts/story_prompt.txt",
                        help="Path to prompt template file")
    parser.add_argument("--emotions_file", type=str, default="prompts/emotions.txt",
                        help="Path to emotions list file")
    parser.add_argument("--topics_file", type=str, default="prompts/topics.txt",
                        help="Path to topics list file")
    
    parser.add_argument("--model_name", type=str, default="swiss-ai/Apertus-8B-Instruct-2509",
                        help="Model name or path")
    
    parser.add_argument("--all_emotions", action="store_true",
                        help="Use all emotions from file")
    parser.add_argument("--emotions_subset", type=int, default=None,
                        help="Use stratified subset of N emotions")
    
    parser.add_argument("--stories_per_prompt", type=int, default=3,
                        help="Number of stories to generate per prompt")
    parser.add_argument("--prompts_per_emotion", type=int, default=3,
                        help="Number of prompts (topics) per emotion")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--stats", type=str, default=None,
                        help="Print statistics for existing stories file")
    
    args = parser.parse_args()
    
    if args.stats:
        print_statistics(args.stats)
        return
    
    generator = EmotionStoryGenerator(
        model_name=args.model_name,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        emotions_file=args.emotions_file,
        topics_file=args.topics_file,
        stories_per_prompt=args.stories_per_prompt,
        prompts_per_emotion=args.prompts_per_emotion,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if args.emotions_subset:
        emotions = select_stratified_emotions(
            generator.all_emotions, 
            args.emotions_subset, 
            seed=args.seed
        )
        print(f"Using stratified subset of {len(emotions)} emotions")
    else:
        emotions = generator.all_emotions
        print(f"Using all {len(emotions)} emotions")
    
    generator.run(
        emotions=emotions,
        topics=generator.all_topics,
        resume=args.resume,
        seed=args.seed,
    )
    
    print_statistics(str(generator.stories_file))
 
 
if __name__ == "__main__":
    main()