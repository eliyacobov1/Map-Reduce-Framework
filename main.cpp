#include "Framework.h"


int initializeJob(const MapReduceClient& client,
                   const InputVec& inputVec, OutputVec& outputVec,
                   int multiThreadLevel)
{
    JobHandle job = startMapReduceJob(client, inputVec,
            outputVec, multiThreadLevel);
    JobState state;
    JobState last_state={UNDEFINED_STAGE,0};
    getJobState(job, &state);

    while (state.stage != REDUCE_STAGE || state.percentage != 100.0)
    {
        if (last_state.stage != state.stage || last_state.percentage != state.percentage){
            printf("stage %d, %f%% \n", state.stage, state.percentage);
        }
        last_state = state;
        getJobState(job, &state);
    }

    printf("stage %d, %f%% \n", state.stage, state.percentage);
    printf("Done!\n");

    closeJobHandle(job);
    return 0;
}

int main(int argc, char** argv)
{

}

