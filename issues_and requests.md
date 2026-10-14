1. Remove the `MIDI in` from the UI - you can keep the logic, but remove it from the UI - the app does not need that
2. When I change AI model to high quality (mrt2_base) it sais "AI unavailable". Probably this model is not installed. When I wil package this app into mac app, will the both models be included? Implement that
3. Add app icon - create one, with guitar in it and add to app
4. I think that we have misunderstanding how this app should work:
- when I want to play guitar, I need ai bass and drums
- when I want to play bass, I need ai rhytm guitar and drums
Right now the app is creating some "AI music" which is not good - this is not an instrument, it just creates some music effects, Is sounds like noise really. Is also creates second drums to the existing ones.
I suggest to have following instrument settings:
- drums on/off with a slider (already existing)
- bass on/off with a slider - enabled only when choosing guitar as playable instrument. There should be no drums or any other effects, only bass
- guitar on/off with a slider - enabled only when choosing bass as playable instrument. There should be no drums or any other effects, only guitar
Add also information what this slider does, is it volume? it is not clear.
Also when I am recording guitar it is saved as "bass.wav", along with "ai_guitar.waw" which was really ai music, and drums. When I record guitar, my recording should be "guitar.wav" along with "bass.wav" (which is ai bass), and "drums.wav". When I record bass, my recording should be "bass.wav" along with "guitar.wav" (which is ai guitar), and "drums.wav".
5. Debug should appear in the right, not in the bottom
6. I don't understand how to use and what does it mean setting "Meter 4/4", or "Meter 3/4" - this app is for begginers mostly and probably they don't know in which meter they will play. How to fix that? I know that this is probably important 
7. You wrote `Or click the Open Folder button in the Record panel inside the JamMate window.` - there is no "open folder" button, there is no record panel, just "record" button and "stems" checkmarks
8. Explain better what does "stems" does - maybe when pointing mouse over it there should explanation appear