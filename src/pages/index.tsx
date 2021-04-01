import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList, } from '@chakra-ui/react'
import { Hero } from 'components/Hero'
import { Container } from 'components/Container'
import NextLink from 'next/link'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

const Index = () => (
  <Container>
    <Hero />
    <LinksRow />
    <Container w="90vw" h="50.6vw" maxW="700px" maxH="393px" mb="3rem">
      <iframe
        width="100%" height="100%"
        src="https://www.youtube.com/embed/ScMzIvxBSi4"
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container>

    <Container w="100%" maxW="44rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontSize="2xl" pb="1rem">Abstract</Heading>
      <Text pb="2rem">
        Over the past year, the emergence of transfer learning with large-scale language models (LM) has led to dramatic performance improvements across a broad range of natural language understanding tasks. However, the size and memory footprint of these large LMs makes them difficult to deploy in many scenarios (e.g. on mobile phones). Recent research points to knowledge distillation as a potential solution, showing that when training data for a given task is abundant, it is possible to distill a large (teacher) LM into a small task-specific (student) network with minimal loss of performance. However, when such data is scarce, there remains a significant performance gap between large pretrained LMs and smaller task-specific models, even when training via distillation
      </Text>

      {/* Example */}
      <Heading fontSize="2xl" pb="1rem">Examples</Heading>
      <img
        src={`${process.env.BASE_PATH || ""}/images/example.png`}
      // width={500}
      // height={500}
      />
      <Text align="center" pt="0.5rem" pb="0.5rem" fontSize="small">This is a caption</Text>

      {/* Another Section */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Another Section</Heading>
      <Text >
        This is another section
      </Text>

      {/* Citation */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;yu2021plenoctrees, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;PlenOctrees for Real - time Rendering of Neural Radiance Fields&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;Alex Yu and Ruilong Li and Matthew Tancik and Hao Li and Ren Ng and Angjoo Kanazawa&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;2021&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;arXiv&#125; <br />
      &#125;
      </Code>

      {/* Related Work */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Related Work</Heading>
      <UnorderedList>
        <ListItem>
          <Text color="blue">
            <NextLink href="#" passHref={true}>
              First paper
            </NextLink>
          </Text>
        </ListItem>
        <ListItem>
          <Text color="blue">
            <NextLink href="#" passHref={true}>
              Second paper
            </NextLink>
          </Text>
        </ListItem>
      </UnorderedList>

      {/* Acknowledgements */}
      <Heading fontSize="2xl" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        We thank xyz for abc...
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index
